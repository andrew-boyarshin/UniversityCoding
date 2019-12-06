#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <deque>
#include <variant>
#include <charconv>
#include <optional>
#include <compare>
#include <string_view>
#include <string>
#include "unique_resource.h"
#include "concurrentqueue.h"
#include "tbb/concurrent_unordered_set.h"
#include <scn/scn.h>
#include <fmt/format.h>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/certify/extensions.hpp>
#include <boost/certify/https_verification.hpp>
#include <boost/algorithm/string.hpp>

#define NOGDI

#include "debug_assert.hpp"
#include "ut.hpp"
#include "plf_nanotimer.h"
#include "dbg.h"
#include <regex>

// cppppack: embed point

struct assert_module : debug_assert::default_handler, debug_assert::set_level<1>
{
};

template <typename T>
using type_decay_basic_t = std::remove_cv_t<std::remove_pointer_t<std::remove_cvref_t<T>>>;

template <typename T>
struct type_decay_ptr
{
    using type = T;
};

template <typename T>
struct type_decay_ptr<std::shared_ptr<T>>
{
    using type = T;
};

template <typename T>
using type_decay_ptr_t = typename type_decay_ptr<T>::type;

template <typename T>
using type_decay_t = type_decay_basic_t<type_decay_ptr_t<type_decay_basic_t<T>>>;

template <typename>
struct is_smart_pointer : std::false_type
{
};

template <typename T, typename D>
struct is_smart_pointer<std::unique_ptr<T, D>> : std::true_type
{
};

template <typename T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type
{
};

template <class T>
inline constexpr bool is_smart_pointer_v = is_smart_pointer<T>::value;

// For convenient std::visit (+deduction guide)
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts ...) -> overloaded<Ts...>;

// Note: only "http", "https", "ws", and "wss" protocols are supported
static const std::regex parse_url{
    R"((([httpsw]{2,5})://)?([^/ :]+)(:(\d+))?(/([^ ?]+)?)?/?\??([^/ ]+\=[^/ ]+)?)",
    std::regex_constants::ECMAScript | std::regex_constants::icase | std::regex_constants::optimize
};

struct parsed_uri final
{
    std::string protocol;
    std::string domain; // only domain must be present
    std::string port;
    std::string resource;
    std::string query; // everything after '?', possibly nothing
    explicit parsed_uri() = default;
    ~parsed_uri() = default;
    parsed_uri(const parsed_uri& other) = default;
    parsed_uri(parsed_uri&& other) noexcept = default;
    parsed_uri& operator=(const parsed_uri& other) = default;
    parsed_uri& operator=(parsed_uri&& other) noexcept = default;
};

parsed_uri parse_uri(const std::string& url)
{
    parsed_uri result;
    auto value_or = [](const std::string& value, std::string&& fallback) -> std::string
    {
        return (value.empty() ? fallback : value);
    };
    std::smatch match;
    if (std::regex_match(url, match, parse_url) && match.size() == 9)
    {
        result.protocol = value_or(boost::algorithm::to_lower_copy(std::string(match[2])), "http");
        result.domain = match[3];
        const auto is_secure_protocol = result.protocol == "https" || result.protocol == "wss";
        result.port = value_or(match[5], is_secure_protocol ? "443" : "80");
        result.resource = value_or(match[6], "/");
        result.query = match[8];
        assert(!result.domain.empty());
    }
    return result;
}

struct job_context
{
    tbb::concurrent_unordered_set<std::string> fetched;
    moodycamel::ConcurrentQueue<std::string> queue;
    std::atomic<std::uint64_t> found_pages;
    std::atomic<std::uint64_t> processed_pages;

    explicit job_context() : found_pages(0), processed_pages(0)
    {
    }

    ~job_context() = default;
    job_context(const job_context& other) = delete;
    job_context(job_context&& other) noexcept = delete;
    job_context& operator=(const job_context& other) = delete;
    job_context& operator=(job_context&& other) noexcept = delete;

    template <typename T>
    void enqueue(T&& url)
    {
        auto [it, inserted] = fetched.insert(url);
        if (inserted)
        {
            found_pages.fetch_add(1, std::memory_order_seq_cst);
            queue.enqueue(url);
        }
    }
};

constexpr auto beast_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0";
constexpr auto beast_accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8";
constexpr auto beast_accept_language = "en-US,en;q=0.5";

namespace beast = boost::beast;
namespace asio = boost::asio;

auto is_ok(const beast::error_code& ec)
{
    // not_connected happens sometimes, so don't bother reporting it.
    return !ec || ec == beast::errc::not_connected;
}

void worker_thread(job_context& context)
{
    // The io_context is required for all I/O
    asio::io_context ioc;

    // The SSL context is required, and holds certificates
    asio::ssl::context ctx(asio::ssl::context::tlsv12_client);

    // Verify the remote server's certificate
    ctx.set_verify_mode(asio::ssl::context::verify_peer);
    ctx.set_default_verify_paths();
    boost::certify::enable_native_https_server_verification(ctx);

    // These objects perform our I/O
    asio::ip::tcp::resolver resolver(ioc);
    asio::ssl::stream<beast::tcp_stream> stream(ioc, ctx);
    std::string url;

    bool items_left;
    do
    {
        // It's important to fence (if the producers have finished) *before* dequeueing
        const auto found = context.found_pages.load(std::memory_order_acquire);
        const auto processed = context.processed_pages.load(std::memory_order_acquire);
        items_left = found != processed;
        while (context.queue.try_dequeue(url))
        {
            items_left = true;

            // This buffer is used for reading and must be persisted
            beast::flat_buffer buffer;
            beast::error_code ec;
            auto status = beast::http::status::unknown;
            do
            {
                if (status == beast::http::status::moved_permanently)
                {
                    // todo: handle redirect
                }

                const auto parsed_url = parse_uri(url);

                dbg(parsed_url.protocol);
                dbg(parsed_url.port);
                dbg(parsed_url.domain);
                dbg(parsed_url.resource);
                dbg(parsed_url.query);

                boost::certify::set_server_hostname(stream, parsed_url.domain);
                boost::certify::sni_hostname(stream, parsed_url.domain);

                // Look up the domain name
                auto const results = resolver.resolve(parsed_url.domain, parsed_url.protocol);

                // Make the connection on the IP address we get from a lookup
                get_lowest_layer(stream).connect(results);
                stream.handshake(asio::ssl::stream_base::client);

                // Set up an HTTP GET request message
                beast::http::request<beast::http::string_body> req;
                req.method(beast::http::verb::get);
                req.target(parsed_url.resource);
                req.set(beast::http::field::host, parsed_url.domain);
                req.set(beast::http::field::accept, beast_accept);
                req.set(beast::http::field::accept_language, beast_accept_language);
                req.set(beast::http::field::user_agent, beast_user_agent);

                // Send the HTTP request to the remote host
                beast::http::write(stream, req);

                // Declare a container to hold the response
                beast::http::response<beast::http::dynamic_body> res;
                buffer.clear();

                // Receive the HTTP response
                beast::http::read(stream, buffer, res, ec);
            }
            while (is_ok(ec));

            // todo: access body ( that was already disposed :( )
            //dbg(res.base());

            // Gracefully close the socket
            stream.shutdown(ec);

            // If we get here then the connection is closed gracefully
            context.processed_pages.fetch_add(1, std::memory_order_release);
        }
    }
    while (items_left);
}

void test_suite();

struct execute_result final
{
    std::uint64_t pages_crawled = 0;
    plf::nanotimer timer;
    explicit execute_result() = default;
    ~execute_result() = default;
    execute_result(const execute_result& other) = delete;
    execute_result(execute_result&& other) noexcept = default;
    execute_result& operator=(const execute_result& other) = delete;
    execute_result& operator=(execute_result&& other) noexcept = default;
};

template <typename Input>
execute_result execute(Input&& in_file)
{
    std::vector<std::thread> threads;
    job_context context;
    execute_result result;

    result.timer.start();

    std::string start_url;
    std::size_t jobs;
    DEBUG_ASSERT(scn::scan(in_file, "{} {}", start_url, jobs), assert_module{});

    context.enqueue(start_url);

    for (decltype(jobs) i = 0; i < jobs; ++i)
    {
        threads.emplace_back(worker_thread, std::ref(context));
    }

    for (auto&& thread : threads)
    {
        thread.join();
    }

    result.pages_crawled = context.processed_pages.load(std::memory_order_seq_cst);
    DEBUG_ASSERT(result.pages_crawled == context.found_pages.load(std::memory_order_seq_cst), assert_module{});

    return result;
}

thread_local const auto close = [](auto fd) { std::fclose(fd); };

int main() // NOLINT(bugprone-exception-escape)
{
    const auto in_file = sr::make_unique_resource_checked(std::fopen("input.txt", "r"), nullptr, close);

    if (in_file.get() == nullptr)
    {
        test_suite();
        return 0;
    }

    const auto out_file = sr::make_unique_resource_checked(std::fopen("output.txt", "w"), nullptr, close);

    if (out_file.get() == nullptr)
    {
        return 1;
    }

    scn::file fin(in_file.get());

    auto result = execute(fin);

    fmt::print(out_file.get(), FMT_STRING("{} {}"), result.pages_crawled, result.timer.get_elapsed_us());

    return 0;
}

#pragma warning( push )
#pragma warning( disable : 26444 )
template <typename T>
void fetch(T&& url)
{
    using namespace boost::ut;
    auto result = execute(scn::make_view(url));
    expect(result.pages_crawled == 1_ul);
    const auto elapsed_ns = result.timer.get_elapsed_ns();
    expect(elapsed_ns > 0._d);
    boost::ut::log << elapsed_ns;
};

void test_suite()
{
    using namespace boost::ut;
    using namespace std::literals;

    "online"_test = []
    {
        "google.com"_test = []
        {
            fetch("https://google.com 1"s);
        };

        "ya.ru"_test = []
        {
            fetch("https://ya.ru 1"s);
        };
    };
}
#pragma warning( pop )
