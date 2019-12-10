#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <deque>
#include <variant>
#include <charconv>
#include <optional>
#include <compare>
#include <string_view>
#include <string>
#include <filesystem>
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
namespace fs = std::filesystem;

namespace magic_enum {
    template <>
    struct enum_range<beast::http::status> {
        static constexpr int min = 0;
        static constexpr int max = 600;
    };
}

struct parsed_url final
{
private:
    std::string protocol_;

public:
    [[nodiscard]] const std::string& protocol() const
    {
        return protocol_;
    }

    [[nodiscard]] std::string& protocol()
    {
        return protocol_;
    }

    template <typename T>
    void protocol(T&& protocol)
    {
        protocol_ = protocol;
        boost::algorithm::to_lower(protocol_);

        deduce_port(false);
    }

    void deduce_port(const bool force = false)
    {
        if (!port.empty() && !force)
        {
            return;
        }

        if (protocol_ == "http" || protocol_ == "ws")
        {
            port = "80";
        }
        if (protocol_ == "https" || protocol_ == "wss")
        {
            port = "443";
        }
    }

    std::string root;
    std::string domain;
    std::string port;
    std::string target;

    explicit parsed_url() = default;
    ~parsed_url() = default;
    parsed_url(const parsed_url& other) = default;
    parsed_url(parsed_url&& other) noexcept = default;
    parsed_url& operator=(const parsed_url& other) = default;
    parsed_url& operator=(parsed_url&& other) noexcept = default;
};

constexpr char hscan_a_pattern[] = R"R(<\s*a[^>]*href\s*=\s*".+"[^>]*>)R";
constexpr char regex_a_pattern[] = R"R(<\s*a[^>]*href\s*=\s*"(.+?)"[^>]*>)R";
constexpr char regex_url_pattern[] = R"R(^(([filehttpsw]{2,5}):)?(/+)(([^\s:/?#\[\]!$&'\(\)*+,;=]+\.[^\s:/?#\[\]!$&'\(\)*+,;=]+)(:(\d+))?)?([^\s?]*?(\?\S*)?)$)R";

#if USE_STD_REGEX
static const std::regex ctre_a_pattern{
    regex_a_pattern,
    std::regex_constants::icase | std::regex_constants::optimize
};
static const std::regex ctre_url_pattern{
    regex_url_pattern,
    std::regex_constants::icase | std::regex_constants::optimize
};

auto regex_match(std::string_view sv, const std::regex& pattern) -> std::optional<std::match_results<decltype(sv)::const_iterator>>
{
    std::match_results<decltype(sv)::const_iterator> match;

    if (std::regex_match(sv.cbegin(), sv.cend(), match, pattern))
    {
        return match;
    }

    return std::nullopt;
}

auto ctre_a_match(std::string_view sv)
{
    return regex_match(sv, ctre_a_pattern);
}

auto ctre_url_match(std::string_view sv)
{
    return regex_match(sv, ctre_url_pattern);
}

template <uint32_t Index, typename T>
auto value_or(T&& match, std::string_view fallback) -> std::string
{
    auto&& value = match[Index];
    return std::string{value.matched ? value.str() : fallback};
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

bool verify_code(const beast::error_code& ec)
{
    if (is_ok(ec))
    {
        return false;
    }

    dbg(ec);
    return true;
}

struct hs_my_context final
{
    std::deque<std::string_view>* matches;
    std::string_view data;

    hs_my_context(std::deque<std::string_view>& matches,
                  std::string_view data)
        : matches(&matches),
          data(data)
    {
    }

    ~hs_my_context() = default;

    hs_my_context(const hs_my_context& other) = delete;
    hs_my_context(hs_my_context&& other) noexcept = delete;
    hs_my_context& operator=(const hs_my_context& other) = delete;
    hs_my_context& operator=(hs_my_context&& other) noexcept = delete;
};

int hs_scan_handler(unsigned int, const unsigned long long from, const unsigned long long to, unsigned int, void* context)
{
    auto* const my_context = static_cast<hs_my_context*>(context);

    my_context->matches->push_back(my_context->data.substr(from, to - from));

    my_context->data.remove_prefix(to);

    return 1;
}

using url_result = std::variant<parsed_url, fs::path>;

std::optional<std::variant<parsed_url, fs::path>> parse_url(const std::string& url, const url_result& previous)
{
    if (const auto match = ctre_url_match(url))
    {
        const auto& m = match.value();
        parsed_url result;

        result.root = value_or<3>(m, "");
        result.domain = value_or<5>(m, "");
        result.port = value_or<7>(m, "");
        result.target = value_or<8>(m, "");

        result.protocol(value_or<2>(m, ""));

        if (result.domain.empty() && result.root.empty() && result.target.empty())
        {
            return std::nullopt;
        }

        if (result.protocol() == "file")
        {
            return std::visit(
                overloaded
                {
                    [](const parsed_url&)
                    {
                        fmt::print(FMT_STRING("Web URL references Local Path, security violation."));
                        return std::nullopt;
                    },
                    [&result](const fs::path& path) -> std::optional<fs::path>
                    {
                        if (result.root.length() < 2 || !result.root.starts_with('/') || !result.root.ends_with('/'))
                        {
                            return std::nullopt;
                        }

                        if (result.root.length() >= 3)
                        {
                            return path.root_path() / result.target;
                        }

                        return fs::relative(fs::path(result.target), path);
                    }
                },
                previous
            );
        }

        return std::visit(
            overloaded
            {
                [&result](const parsed_url& url)
                {
                    return result;
                },
                [](const fs::path&) -> std::optional<fs::path>
                {
                    fmt::print(FMT_STRING("Local Path references Web URL, security violation."));
                    return std::nullopt;
                }
            },
            previous
        );
    }

    return std::nullopt;
}

std::optional<std::string> parse_a(const std::string_view& a)
{
    if (const auto match = ctre_a_match(a))
    {
        const auto& m = match.value();

        auto result = value_or<1>(m, "");
        if (result.empty())
        {
            return std::nullopt;
        }

        return result;
    }

    return std::nullopt;
}

void parse_content(job_context& context, hs_scratch_t* scratch, std::string& body)
{
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

    for (auto&& match : matches)
    {
        auto a_opt = parse_a(match);
        if (!a_opt)
        {
            fmt::print(FMT_STRING("* \"{}\" couldn't be parsed as <A /> element\n"), match);
            continue;
        }

        context.enqueue(a_opt.value());
    }
}

bool fetch_web(job_context& context, asio::ssl::context& ctx, asio::ip::tcp::resolver& resolver, asio::io_context& ioc,
               std::string& url_ref, hs_scratch_t* scratch, const parsed_url& url, beast::flat_buffer& buffer)
{
    const auto protocol = url.protocol();

    dbg(protocol);
    dbg(url.port);
    dbg(url.domain);
    dbg(url.target);

    asio::ssl::stream<beast::tcp_stream> stream(ioc, ctx);

    beast::error_code ec;

    [[maybe_unused]] sr::scope_exit stream_shutdown{
        [&stream, &ec]
        {
            stream.shutdown(ec);
        }
    };

    std::optional<boost::system::error_code> timer_result;
    asio::deadline_timer timer(ioc);
    timer.expires_from_now(boost::posix_time::seconds(5), ec);

    if (verify_code(ec))
    {
        return false;
    }

    timer.async_wait(
        [&timer_result, &stream](const boost::system::error_code& error)
        {
            if (error != asio::error::basic_errors::operation_aborted)
            {
                beast::get_lowest_layer(stream).cancel();
            }

            timer_result = error;
        }
    );

    boost::certify::set_server_hostname(stream, url.domain, ec);

    if (verify_code(ec))
    {
        return false;
    }

    boost::certify::sni_hostname(stream, url.domain, ec);

    if (verify_code(ec))
    {
        return false;
    }

    // Look up the domain name
    auto const results = resolver.resolve(url.domain, protocol, ec);

    if (verify_code(ec))
    {
        return false;
    }

    // Make the connection on the IP address we get from a lookup
    get_lowest_layer(stream).connect(results, ec);

    if (verify_code(ec))
    {
        return false;
    }

    stream.handshake(asio::ssl::stream_base::client, ec);

    if (verify_code(ec))
    {
        return false;
    }

    // Set up an HTTP GET request message
    beast::http::request<beast::http::string_body> req;
    req.method(beast::http::verb::get);
    req.target(url.target);
    req.set(beast::http::field::host, url.domain);
    req.set(beast::http::field::accept, beast_accept);
    req.set(beast::http::field::accept_language, beast_accept_language);
    req.set(beast::http::field::user_agent, beast_user_agent);

    // Send the HTTP request to the remote host
    beast::http::write(stream, req, ec);

    if (verify_code(ec))
    {
        return false;
    }

    // Declare a container to hold the response
    beast::http::response<beast::http::string_body> res;
    buffer.clear();

    // Receive the HTTP response
    beast::http::read(stream, buffer, res, ec);

    if (verify_code(ec))
    {
        return false;
    }

    auto result = res.base().result();
    switch (result)
    {
        case beast::http::status::ok:
        {
            parse_content(context, scratch, res.body());

            return false;
        }

        case beast::http::status::found:
        case beast::http::status::moved_permanently:
        case beast::http::status::permanent_redirect:
        case beast::http::status::temporary_redirect:
        {
            const auto location = res.base().at(beast::http::field::location);
            url_ref.assign(location.data(), location.size());
            return true;
        }

        default:
        {
            std::string v{magic_enum::enum_name<beast::http::status>(result)};
            dbg(v);
            return false;
        }
    }
}

thread_local const auto close = [](auto fd) { std::fclose(fd); };

void fetch_file(job_context& context, const fs::path& path, hs_scratch_t* scratch)
{
    const auto in_file = sr::make_unique_resource_checked(_wfopen(path.c_str(), L"r"), nullptr, close);

    if (in_file.get() == nullptr)
    {
        return;
    }

    const auto in_view = scn::file(in_file.get());

    const auto file_size = fs::file_size(path);
    std::string buffer;
    buffer.reserve(file_size);

    auto it = std::back_inserter(buffer);

    auto range = in_view.wrap();
    scn::read_into(range, it, file_size);

    parse_content(context, scratch, buffer);
}

void fetch(job_context& context, asio::ssl::context& ctx, asio::ip::tcp::resolver& resolver, asio::io_context& ioc,
               std::string url, hs_scratch_t* scratch)
{
    // This buffer is used for reading and must be persisted
    beast::flat_buffer buffer;
    auto new_target = true;
    while (new_target)
    {
        const auto parsed_url_opt = parse_url(url);

        if (!parsed_url_opt)
        {
            fmt::print(FMT_STRING("Failed to parse \"{}\" as URL\n"), url);
            return;
        }

        const auto& parsed_url_variant = parsed_url_opt.value();

        new_target = std::visit(
            overloaded
            {
                [&context, &ctx, &resolver, &ioc, &url_ref = url, scratch, &buffer](const parsed_url& url)
                {
                    return fetch_web(
                        context, ctx, resolver, ioc, url_ref, scratch, url, buffer
                    );
                },
                [&context, scratch](const fs::path& path)
                {
                    fetch_file(context, path, scratch);
                    return false;
                }
            },
            parsed_url_variant
        );
    }
}

void worker_thread(job_context& context)
{
    // The io_context is required for all I/O
    asio::io_context ioc;

    // The SSL context is required, and holds certificates
    asio::ssl::context ctx(asio::ssl::context::tls_client);
    ctx.set_options(asio::ssl::context_base::default_workarounds);

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
            [[maybe_unused]] sr::scope_exit inc_processed_pages{
                [&context]
                {
                    context.processed_pages.fetch_add(1, std::memory_order_release);
                }
            };

            items_left = true;

            fetch(context, ctx, resolver, ioc, url, scratch_thread);
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
    if (hs_compile(hscan_a_pattern, HS_FLAG_CASELESS | HS_FLAG_SOM_LEFTMOST, HS_MODE_BLOCK, nullptr, &database, &compile_err) != HS_SUCCESS) {
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
