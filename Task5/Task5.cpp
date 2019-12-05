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
#include <boost/beast/version.hpp>
#include <boost/asio/strand.hpp>
#include <boost/algorithm/string.hpp>

#define NOGDI

#include "debug_assert.hpp"
#include "ut.hpp"
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

std::vector<std::thread> threads;
tbb::concurrent_unordered_set<std::string> fetched;
moodycamel::ConcurrentQueue<std::string> queue;
std::atomic<std::uint64_t> found_pages(0);
std::atomic<std::uint64_t> processed_pages(0);

namespace beast = boost::beast; // from <boost/beast.hpp>
namespace http = beast::http; // from <boost/beast/http.hpp>
namespace net = boost::asio; // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>

// Report a failure
void
fail(beast::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// Performs an HTTP GET and prints the response
class session : public std::enable_shared_from_this<session>
{
    tcp::resolver resolver_;
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_; // (Must persist between reads)
    http::request<http::empty_body> req_;
    http::response<http::string_body> res_;

public:
    // Objects are constructed with a strand to
    // ensure that handlers do not execute concurrently.
    explicit session(net::io_context& ioc)
        : resolver_(net::make_strand(ioc))
          , stream_(net::make_strand(ioc))
    {
    }

    // Start the asynchronous operation
    void run(
        const std::string& url
    )
    {
        const auto parsed_url = parse_uri(url);

        // Set up an HTTP GET request message
        req_.method(http::verb::get);
        req_.target(parsed_url.resource);
        req_.set(http::field::host, parsed_url.domain);
        req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

        // Look up the domain name
        resolver_.async_resolve(
            parsed_url.domain,
            parsed_url.protocol,
            beast::bind_front_handler(
                &session::on_resolve,
                shared_from_this()));
    }

    void on_resolve(
        beast::error_code ec,
        tcp::resolver::results_type results
    )
    {
        if (ec)
            return fail(ec, "resolve");

        // Set a timeout on the operation
        stream_.expires_after(std::chrono::seconds(30));

        // Make the connection on the IP address we get from a lookup
        stream_.async_connect(
            results,
            beast::bind_front_handler(
                &session::on_connect,
                shared_from_this()));
    }

    void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type)
    {
        if (ec)
            return fail(ec, "connect");

        // Set a timeout on the operation
        stream_.expires_after(std::chrono::seconds(30));

        // Send the HTTP request to the remote host
        http::async_write(stream_, req_,
                          beast::bind_front_handler(
                              &session::on_write,
                              shared_from_this()));
    }

    void on_write(
        beast::error_code ec,
        std::size_t bytes_transferred
    )
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return fail(ec, "write");

        // Receive the HTTP response
        http::async_read(stream_, buffer_, res_,
                         beast::bind_front_handler(
                             &session::on_read,
                             shared_from_this()));
    }

    void on_read(
        beast::error_code ec,
        std::size_t bytes_transferred
    )
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return fail(ec, "read");

        // Write the message to standard out
        std::cout << res_ << std::endl;

        // Gracefully close the socket
        stream_.socket().shutdown(tcp::socket::shutdown_both, ec);

        // not_connected happens sometimes so don't bother reporting it.
        if (ec && ec != beast::errc::not_connected)
            return fail(ec, "shutdown");

        // If we get here then the connection is closed gracefully
    }
};

void worker_thread()
{
    std::string item;
    for (int j = 0; j != 20; ++j)
    {
        if (queue.try_dequeue(item))
        {
            //++dequeued[item];
        }
    }

    for (int j = 0; j != 10; ++j)
    {
        queue.enqueue("Test");
    }
}


void test_suite();

template <typename Input, typename Output>
void execute(Input&& in_file, Output&& out_file)
{
    std::string start_url;
    std::size_t jobs;
    DEBUG_ASSERT(scn::scan(in_file, "{} {}", start_url, jobs), assert_module{});

    std::uint64_t pages = 1;
    std::uint64_t time_ms = 100;
    if constexpr (std::is_pointer_v<std::remove_cvref_t<Output>>)
    {
        fmt::print(out_file, FMT_STRING("{} {}"), pages, time_ms);
    }
    else
    {
        fmt::format_to(out_file, FMT_STRING("{} {}"), pages, time_ms);
    }

    // The io_context is required for all I/O
    net::io_context ioc;

    auto ptr = std::make_shared<session>(ioc);
    // Launch the asynchronous operation
    ptr->run("https://google.com");

    // Run the I/O service. The call will return when
    // the get operation is complete.
    ioc.run();
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

    execute(fin, out_file.get());

    return 0;
}

#pragma warning( push )
#pragma warning( disable : 26444 )
void test_suite()
{
    using namespace boost::ut;
    using namespace std::literals;

    "examples"_test = []
    {
        "1"_test = []
        {
            fmt::memory_buffer buf;
            auto input = "https://google.com 1"s;
            execute(scn::make_view(input), buf);
            expect(eq(fmt::to_string(buf), "1 100"sv));
        };
    };
}
#pragma warning( pop )
