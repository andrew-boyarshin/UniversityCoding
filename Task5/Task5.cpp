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
#define USE_CTRE 0
#define USE_STD_REGEX 1

#include "magic_enum.hpp"
#include "debug_assert.hpp"
#include "ut.hpp"
#include "plf_nanotimer.h"
#include "dbg.h"
#include "hs.h"
#if USE_STD_REGEX
#include <regex>
#elif USE_CTRE
#include "ctre.hpp"
#endif

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

namespace beast = boost::beast;
namespace asio = boost::asio;

namespace magic_enum {
    template <>
    struct enum_range<beast::http::status> {
        static constexpr int min = 0;
        static constexpr int max = 600;
    };
}

struct parsed_uri final
{
    std::string protocol;
    std::string domain;
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

constexpr char regex_a_pattern[] = R"R(<\s*a[^>]*href\s*=\s*"(.+?)"[^>]*>)R";
constexpr char regex_url_pattern[] = R"R((([filehttpsw]{2,5})://)?([^/ :]+)(:(\d+))?(/([^ ?]+)?)?/?\??(\S+)?)R";

#if USE_STD_REGEX
static const std::regex ctre_a_pattern{
    regex_a_pattern,
    std::regex_constants::icase | std::regex_constants::optimize
};
static const std::regex ctre_url_pattern{
    regex_url_pattern,
    std::regex_constants::icase | std::regex_constants::optimize
};

auto ctre_a_match(std::string_view sv) noexcept {
    std::match_results<decltype(sv)::const_iterator> match;
    std::regex_match(sv.cbegin(), sv.cend(), match, ctre_a_pattern);
    return match;
}

auto ctre_url_match(std::string_view sv) noexcept {
    std::match_results<decltype(sv)::const_iterator> match;
    std::regex_match(sv.cbegin(), sv.cend(), match, ctre_url_pattern);
    return match;
}

template <uint32_t Index, typename T>
auto value_or(T&& match, std::string_view fallback) -> std::string_view
{
    auto&& value = match.get<Index>();
    return value ? value.to_view() : fallback;
}
#elif USE_CTRE
static constexpr auto ctre_a_pattern = ctll::fixed_string{  };
static constexpr auto ctre_url_pattern = ctll::fixed_string{  };

constexpr auto ctre_a_match(std::string_view sv) noexcept {
    return ctre::match<ctre_a_pattern>(sv);
}

constexpr auto ctre_url_match(std::string_view sv) noexcept {
    return ctre::match<ctre_url_pattern>(sv);
}

template <uint32_t Index, typename T>
auto value_or(T&& match, std::string_view fallback) -> std::string_view
{
    auto&& value = match.get<Index>();
    return value ? value.to_view() : fallback;
}
#endif

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

hs_database_t* database = nullptr;
hs_scratch_t* scratch_prototype = nullptr;

auto is_ok(const beast::error_code& ec)
{
    // not_connected happens sometimes, so don't bother reporting it.
    return !ec || ec == beast::errc::not_connected;
}

struct hs_my_context final
{
    std::deque<std::string_view>* matches;
    std::string_view data;

    hs_my_context(std::deque<std::string_view>& matches,
                  std::string_view data)
        : matches(&matches),
          data(std::move(data))
    {
    }

    ~hs_my_context() = default;

    hs_my_context(const hs_my_context& other) = delete;
    hs_my_context(hs_my_context&& other) noexcept = delete;
    hs_my_context& operator=(const hs_my_context& other) = delete;
    hs_my_context& operator=(hs_my_context&& other) noexcept = delete;
};

int hs_scan_handler(unsigned int, unsigned long long from, unsigned long long to, unsigned int, void* context)
{
    auto* const my_context = static_cast<hs_my_context*>(context);

    my_context->matches->push_back(my_context->data.substr(from, to - from));

    my_context->data.remove_prefix(to);

    return 1;
}

std::optional<parsed_uri> parse_uri(const std::string& url)
{
    if (auto m = ctre_url_match(url))
    {
        parsed_uri result;

        result.protocol = value_or<2>(m, "http");
        boost::algorithm::to_lower(result.protocol);
        result.domain = value_or<3>(m, "");
        const auto is_secure_protocol = result.protocol == "https" || result.protocol == "wss";
        result.port = value_or<5>(m, is_secure_protocol ? "443" : "80");
        result.resource = value_or<6>(m, "/");
        result.query = value_or<8>(m, "");
        DEBUG_ASSERT(!result.domain.empty(), assert_module{});

        return result;
    }

    return std::nullopt;
}

std::optional<std::string> parse_a(const std::string_view& a)
{
    if (auto m = ctre_a_match(a))
    {
        std::string result;
        result = value_or<1>(m, "");
        if (result.empty())
        {
            return std::nullopt;
        }

        return result;
    }

    return std::nullopt;
}

void fetch_web(job_context& context, asio::ip::tcp::resolver& resolver, asio::ssl::stream<beast::tcp_stream>& stream, std::string url, hs_scratch_t* scratch)
{
    beast::error_code ec;
    // This buffer is used for reading and must be persisted
    beast::flat_buffer buffer;
    do
    {
        const auto parsed_url_opt = parse_uri(url);

        if (!parsed_url_opt)
        {
            fmt::print(FMT_STRING("Failed to parse \"{}\" as URL\n"), url);
            return;
        }

        const auto& parsed_url = parsed_url_opt.value();

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
        beast::http::response<beast::http::string_body> res;
        buffer.clear();

        // Receive the HTTP response
        beast::http::read(stream, buffer, res, ec);

        auto result = res.base().result();
        switch (result)
        {
            case beast::http::status::ok:
            {
                auto& body = res.body();
                std::deque<std::string_view> matches;

                hs_my_context my_context{matches, body.data()};

                while (!my_context.data.empty())
                {
                    const auto ret = hs_scan(
                        database, my_context.data.data(), my_context.data.length(), 0, scratch,
                        hs_scan_handler, &my_context
                    );

                    if (ret != HS_SUCCESS && ret != HS_SCAN_TERMINATED)
                    {
                        fmt::print(FMT_STRING("ERROR: Unable to scan input buffer. Exiting.\n"));
                        return;
                    }

                    if (ret == HS_SUCCESS)
                    {
                        break;
                    }
                }

                for (auto && match : matches)
                {
                    auto a_opt = parse_a(match);
                    if (!a_opt)
                    {
                        fmt::print(FMT_STRING("* \"{}\" couldn't be parsed as <A /> element\n"), match);
                        continue;
                    }

                    context.enqueue(a_opt.value());
                }

                return;
            }
            case beast::http::status::moved_permanently:
            case beast::http::status::permanent_redirect:
            case beast::http::status::temporary_redirect:
            {
                const auto location = res.base().at(beast::http::field::location);
                url.assign(location.data(), location.size());
                break;
            }
            default:
            {
                std::string v{magic_enum::enum_name<beast::http::status>(result)};
                dbg(v);
                return;
            }
        }
    }
    while (is_ok(ec));
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

    hs_scratch_t* scratch_thread = nullptr;

    if (hs_clone_scratch(scratch_prototype, &scratch_thread) != HS_SUCCESS) {
        fmt::print(FMT_STRING("hs_clone_scratch failed!\n"));
        std::exit(1);
    }

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

            asio::ssl::stream<beast::tcp_stream> stream(ioc, ctx);

            fetch_web(context, resolver, stream, url, scratch_thread);

            beast::error_code ec;
            // Gracefully close the socket
            stream.shutdown(ec);

            // If we get here then the connection is closed gracefully
            context.processed_pages.fetch_add(1, std::memory_order_release);
        }
    }
    while (items_left);

    hs_free_scratch(scratch_thread);
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

void initialize_regex()
{
    hs_compile_error_t* compile_err;
    if (hs_compile(R"(<\s*a[^>]*href\s*=\s*".+"[^>]*>)", HS_FLAG_CASELESS | HS_FLAG_SOM_LEFTMOST, HS_MODE_BLOCK, nullptr, &database, &compile_err) != HS_SUCCESS) {
        fmt::print(FMT_STRING("ERROR: Unable to compile pattern: {}\n"), compile_err->message);
        hs_free_compile_error(compile_err);
        std::exit(1);
    }

    if (hs_alloc_scratch(database, &scratch_prototype) != HS_SUCCESS) {
        fmt::print(FMT_STRING("ERROR: Unable to allocate scratch space. Exiting.\n"));
        hs_free_database(database);
        std::exit(1);
    }
}

void destruct_regex()
{
    hs_free_scratch(scratch_prototype);
    scratch_prototype = nullptr;
    hs_free_database(database);
    database = nullptr;
}

template <typename Input>
execute_result execute(Input&& in_file)
{
    sr::scope_exit v{ destruct_regex };
    initialize_regex();

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
