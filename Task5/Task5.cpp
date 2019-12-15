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
#include <mio.hpp>
#include <scn/scn.h>
#include <fmt/format.h>
#include <boost/functional/hash.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ssl/error.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/certify/extensions.hpp>
#include <boost/certify/https_verification.hpp>
#include <boost/algorithm/string.hpp>

#define USE_CTRE 0
#define USE_STD_REGEX 1

#include "magic_enum.hpp"
#include "debug_assert.hpp"
#include "ut.hpp"
#include "plf_nanotimer.h"
#include "dbg.h"
#include "hs.h"
#include "igor.hpp"
#if USE_STD_REGEX
#include <regex>
#elif USE_CTRE
#include "ctre.hpp"
#endif

// cppppack: embed point

struct job_context;

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

struct parsed_uri final
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

        std::visit(
            [this](auto&& value) { value.deduce_port(protocol_, false); },
            variant
        );
    }

    struct parsed_link
    {
        virtual ~parsed_link() = default;
        virtual void deduce_port(const std::string& protocol, bool force = false) = 0;
        [[nodiscard]] virtual bool good() const = 0;
    };

    struct parsed_url final : parsed_link
    {
        std::string domain;
        std::string port;
        std::string target;

        parsed_url(const std::string& protocol, const std::string_view domain, const std::string_view port, const std::string_view target)
            : domain(domain),
              port(port),
              target(target)
        {
            deduce_port(protocol, false);
        }

        ~parsed_url() override = default;

        parsed_url(const parsed_url& other) = default;
        parsed_url(parsed_url&& other) noexcept = default;
        parsed_url& operator=(const parsed_url& other) = default;
        parsed_url& operator=(parsed_url&& other) noexcept = default;

        void deduce_port(const std::string& protocol, const bool force = false) override
        {
            if (!port.empty() && !force)
            {
                return;
            }

            if (protocol == "http" || protocol == "ws")
            {
                port = "80";
            }

            if (protocol == "https" || protocol == "wss")
            {
                port = "443";
            }
        }

        static std::size_t hash_value(const parsed_url& obj)
        {
            std::size_t seed = 0x7CCB4EE8;
            boost::hash_combine(seed, obj.domain);
            boost::hash_combine(seed, obj.port);
            boost::hash_combine(seed, obj.target);
            return seed;
        }

        friend std::size_t hash_value(const parsed_url& obj)
        {
            return hash_value(obj);
        }

        [[nodiscard]] bool good() const override
        {
            return !domain.empty() || !target.empty();
        }

        friend bool operator==(const parsed_url& lhs, const parsed_url& rhs)
        {
            return lhs.domain == rhs.domain
                && lhs.port == rhs.port
                && lhs.target == rhs.target;
        }

        friend bool operator!=(const parsed_url& lhs, const parsed_url& rhs)
        {
            return !(lhs == rhs);
        }
    };

    struct parsed_local final : parsed_link
    {
        fs::path path;

        explicit parsed_local(fs::path path)
            : path(std::move(path))
        {
        }

        ~parsed_local() override = default;

        parsed_local(const parsed_local& other) = default;
        parsed_local(parsed_local&& other) noexcept = default;
        parsed_local& operator=(const parsed_local& other) = default;
        parsed_local& operator=(parsed_local&& other) noexcept = default;

        void deduce_port(const std::string&, bool = false) override
        {
        }

        [[nodiscard]] bool good() const override
        {
            return !path.empty() && fs::exists(path);
        }

        static std::size_t hash_value(const parsed_local& obj)
        {
            std::size_t seed = 0x6D822F9D;
            boost::hash_combine(seed, obj.path);
            return seed;
        }

        friend std::size_t hash_value(const parsed_local& obj)
        {
            return hash_value(obj);
        }

        friend bool operator==(const parsed_local& lhs, const parsed_local& rhs)
        {
            return lhs.path == rhs.path;
        }

        friend bool operator!=(const parsed_local& lhs, const parsed_local& rhs)
        {
            return !(lhs == rhs);
        }
    };

    struct parsed_placeholder final : parsed_link
    {
        explicit parsed_placeholder() = default;
        ~parsed_placeholder() = default;
        parsed_placeholder(const parsed_placeholder& other) = default;
        parsed_placeholder(parsed_placeholder&& other) noexcept = default;
        parsed_placeholder& operator=(const parsed_placeholder& other) = default;
        parsed_placeholder& operator=(parsed_placeholder&& other) noexcept = default;

        void deduce_port(const std::string&, bool = false) override
        {
        }

        [[nodiscard]] bool good() const override
        {
            return false;
        }

        static std::size_t hash_value(const parsed_placeholder&)
        {
            return 0x105FAB96;
        }

        friend std::size_t hash_value(const parsed_placeholder& obj)
        {
            return hash_value(obj);
        }

        friend bool operator==(const parsed_placeholder&, const parsed_placeholder&)
        {
            return true;
        }

        friend bool operator!=(const parsed_placeholder& lhs, const parsed_placeholder& rhs)
        {
            return !(lhs == rhs);
        }
    };

    std::variant<parsed_url, parsed_local, parsed_placeholder> variant;

    explicit parsed_uri(fs::path path)
        : protocol_{"file"},
          variant{std::in_place_type_t<parsed_local>{}, std::move(path)}
    {
    }

    parsed_uri(const std::string_view protocol, std::string_view domain, std::string_view port, std::string_view target)
        : protocol_{protocol},
          variant{std::in_place_type_t<parsed_url>{}, protocol_, domain, port, target}
    {
    }

    ~parsed_uri() = default;

    parsed_uri(const parsed_uri& other) = default;
    parsed_uri(parsed_uri&& other) noexcept = default;
    parsed_uri& operator=(const parsed_uri& other) = default;
    parsed_uri& operator=(parsed_uri&& other) noexcept = default;

    [[nodiscard]] bool good() const
    {
        return std::visit(
            [](auto&& value) { return value.good(); },
            variant
        );
    }

    static std::size_t hash_value(const parsed_uri& obj);

    friend std::size_t hash_value(const parsed_uri& obj)
    {
        return hash_value(obj);
    }

    friend bool operator==(const parsed_uri& lhs, const parsed_uri& rhs)
    {
        return lhs.protocol_ == rhs.protocol_
            && lhs.variant == rhs.variant;
    }

    friend bool operator!=(const parsed_uri& lhs, const parsed_uri& rhs)
    {
        return !(lhs == rhs);
    }

    friend void worker_thread(job_context& context);

private:
    explicit parsed_uri() : variant{std::in_place_type_t<parsed_placeholder>{}}
    {
    }
};

namespace std
{
    template<>
    struct hash<parsed_uri::parsed_url>
    {
        std::size_t operator()(parsed_uri::parsed_url const& s) const noexcept
        {
            return parsed_uri::parsed_url::hash_value(s);
        }
    };
    template<>
    struct hash<parsed_uri::parsed_local>
    {
        std::size_t operator()(parsed_uri::parsed_local const& s) const noexcept
        {
            return parsed_uri::parsed_local::hash_value(s);
        }
    };
    template<>
    struct hash<parsed_uri::parsed_placeholder>
    {
        std::size_t operator()(parsed_uri::parsed_placeholder const& s) const noexcept
        {
            return parsed_uri::parsed_placeholder::hash_value(s);
        }
    };
    template<>
    struct hash<parsed_uri>
    {
        std::size_t operator()(parsed_uri const& s) const noexcept
        {
            return parsed_uri::hash_value(s);
        }
    };
}

namespace tbb
{
    template <>
    class tbb_hash<parsed_uri>
    {
    public:
        std::size_t operator()(parsed_uri const& s) const noexcept
        {
            return parsed_uri::hash_value(s);
        }
    };
}

std::size_t parsed_uri::hash_value(const parsed_uri& obj)
{
    std::size_t seed = 0x48EB80B7;
    boost::hash_combine(seed, obj.protocol_);
    boost::hash_combine(seed, obj.variant);
    return seed;
}

constexpr char hscan_a_pattern[] = R"R(<\s*a[^>]*href\s*=\s*".+"[^>]*>)R";
constexpr char regex_a_pattern[] = R"R(<\s*a[^>]*href\s*=\s*"(.+?)"[^>]*>)R";
constexpr char regex_url_pattern[] = R"R(^(([https]{4,5}):)?(/*)(([^\s:/?#\[\]!$&'\(\)*+,;=]+\.[^\s:/?#\[\]!$&'\(\)*+,;=]+)(:(\d+))?)?([^\s?]*(\?\S*)?)$)R";
constexpr char regex_local_pattern[] = R"R(^(([filehttpsw]{2,5}):)?(/+)(([^\s:/?#\[\]!$&'\(\)*+,;=]+\.[^\s:/?#\[\]!$&'\(\)*+,;=]+)(:(\d+))?)?([^\s?]*?(\?\S*)?)$)R";

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
auto value_of(T&& match) -> std::string
{
    auto&& value = match[Index];

    if (!value.matched)
    {
        return "";
    }

    return value.str();
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

struct job_context final
{
    tbb::concurrent_unordered_set<parsed_uri> fetched;
    moodycamel::ConcurrentQueue<parsed_uri> queue;
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

struct referrer_tag
{
};

inline constexpr auto referrer = igor::named_argument<referrer_tag>{};

constexpr std::array<std::string_view, 3> protocol_prefixes = {
    "http://",
    "https://",
    "file://"
};

template <typename... Args>
std::optional<parsed_uri> parse_url(std::string_view url, Args&&... args)
{
    if (url.starts_with('#'))
    {
        return std::nullopt;
    }

    igor::parser p{ args... };
    auto& referrer = p(::referrer);

    auto root_slash_count = 0;
    std::string_view protocol;
    for (auto && prefix : protocol_prefixes)
    {
        if (url.starts_with(prefix))
        {
            protocol = url.substr(0, url.find(':'));
            root_slash_count = -2;
            url.remove_prefix(protocol.length());
            break;
        }
    }

    constexpr auto have_referrer = p.has(::referrer);

    if constexpr (have_referrer)
    {
        if (protocol.empty() && referrer.good())
        {
            protocol = referrer.protocol();
        }
    }

    if (protocol.empty())
    {
        return std::nullopt;
    }

    const auto local = protocol == "file";

    if (url.front() == ':')
    {
        url.remove_prefix(1);
    }

    while (!url.empty() && url.front() == '/')
    {
        url.remove_prefix(1);
        root_slash_count++;
    }

    if (local)
    {
        fs::path path{ url };

        if constexpr (have_referrer)
        {
            if (!path.is_absolute() || !fs::exists(path))
            {
                if (root_slash_count > 0)
                {
                    std::visit(
                        overloaded
                        {
                            [&path](const parsed_uri::parsed_local& referrer_local)
                            {
                                path = referrer_local.path.root_path() / path;
                            },
                            [](auto&&)
                            {
                                fmt::print(FMT_STRING("URI target zone mismatch"));
                            },
                        },
                        referrer.variant
                    );
                }
                else
                {
                    std::visit(
                        overloaded
                        {
                            [&path](const parsed_uri::parsed_local& referrer_local)
                            {
                                path = referrer_local.path.parent_path() / path;
                            },
                            [](auto&&)
                            {
                                fmt::print(FMT_STRING("URI target zone mismatch"));
                            },
                        },
                        referrer.variant
                    );
                }
            }
        }

        path = fs::absolute(path);

        if (!fs::exists(path))
        {
            return std::nullopt;
        }

        return parsed_uri{path};
    }

    const auto port_colon_pos = url.find(':');

    std::string_view domain;
    std::string_view port;

    constexpr auto npos = decltype(url)::npos;
    if (port_colon_pos != npos)
    {
        const auto next_part_pos = url.find_first_of("/?");

        // "//blog.google/q?v=https://test.com:443/a"
        auto next_part_ok = next_part_pos == npos || next_part_pos > port_colon_pos;
        if (next_part_ok)
        {
            domain = url.substr(0, port_colon_pos);
            url.remove_prefix(port_colon_pos + 1);
            const auto port_end = url.find_first_not_of("0123456789");
            port = url.substr(0, port_end);
            url.remove_prefix(std::min(port_end, url.size()));
        }
    }

    if (domain.empty())
    {
        if (root_slash_count == 0)
        {
            const auto domain_end = url.find_first_of("/?");
            domain = url.substr(0, domain_end);
            url.remove_prefix(std::min(domain_end, url.size()));
        }
        else if (root_slash_count > 0)
        {
            if constexpr (have_referrer)
            {
                std::visit(
                    overloaded
                    {
                        [&domain, &port](const parsed_uri::parsed_url& referrer_url)
                        {
                            domain = referrer_url.domain;
                            port = referrer_url.port;
                        },
                        [](auto&&)
                        {
                            fmt::print(FMT_STRING("URI target zone mismatch"));
                        },
                    },
                    referrer.variant
                );
            }
        }
    }

    auto target = url;

    if (!domain.empty() || root_slash_count >= 1 || !target.empty())
    {
        if (target.empty())
        {
            target = "/";
        }

        return parsed_uri
        {
            protocol,
            domain,
            port,
            target
        };
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

void parse_content(job_context& context, hs_scratch_t* scratch, std::string_view body, const parsed_uri& url)
{
    std::deque<std::string_view> matches;

    hs_my_context my_context{matches, body};

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

        auto& href = a_opt.value();
        auto parsed_href = parse_url(href, referrer = url);

        if (!parsed_href)
        {
            fmt::print(FMT_STRING("* \"{}\" couldn't be parsed as a URI address\n"), href);
            continue;
        }

        context.enqueue(parsed_href.value());
    }
}

void fetch_web(job_context& context, asio::ssl::context& ctx, asio::ip::tcp::resolver& resolver, asio::io_context& ioc,
               hs_scratch_t* scratch, const parsed_uri& uri, beast::flat_buffer& buffer)
{
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
        return;
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

    auto& url = std::get<parsed_uri::parsed_url>(uri.variant);

    boost::certify::set_server_hostname(stream, url.domain, ec);

    if (verify_code(ec))
    {
        return;
    }

    boost::certify::sni_hostname(stream, url.domain, ec);

    if (verify_code(ec))
    {
        return;
    }

    // Look up the domain name
    auto const results = resolver.resolve(url.domain, uri.protocol(), ec);

    if (verify_code(ec))
    {
        return;
    }

    // Make the connection on the IP address we get from a lookup
    get_lowest_layer(stream).connect(results, ec);

    if (verify_code(ec))
    {
        return;
    }

    stream.handshake(asio::ssl::stream_base::client, ec);

    if (verify_code(ec))
    {
        return;
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
        return;
    }

    // Declare a container to hold the response
    beast::http::response<beast::http::string_body> res;
    buffer.clear();

    // Receive the HTTP response
    beast::http::read(stream, buffer, res, ec);

    if (verify_code(ec))
    {
        return;
    }

    auto result = res.base().result();
    switch (result)
    {
        case beast::http::status::ok:
        {
            parse_content(context, scratch, res.body(), uri);

            break;
        }

        case beast::http::status::found:
        case beast::http::status::moved_permanently:
        case beast::http::status::permanent_redirect:
        case beast::http::status::temporary_redirect:
        {
            const auto location = res.base().at(beast::http::field::location);
            const auto target_url = parse_url(location, referrer = uri);
            if (!target_url)
            {
                // todo
                break;
            }
            context.enqueue(target_url.value());
            break;
        }

        default:
        {
            std::string v{magic_enum::enum_name<beast::http::status>(result)};
            dbg(v);
            break;
        }
    }
}

thread_local const auto close = [](auto fd) { std::fclose(fd); };

void fetch_file(job_context& context, const parsed_uri& uri, hs_scratch_t* scratch)
{
    const auto& uri_local = std::get<parsed_uri::parsed_local>(uri.variant);
    const auto& path = uri_local.path;

    std::error_code error;
    const auto mmap = ::mio::make_mmap_source(path.native(), error);

    if (error)
    {
        fmt::print(FMT_STRING("File \"{}\" cannot be opened ({}).\n"), path.string(), error.message());
        return;
    }

    parse_content(context, scratch, {mmap.data(), mmap.length()}, uri);
}

void fetch(job_context& context, asio::ssl::context& ctx, asio::ip::tcp::resolver& resolver, asio::io_context& ioc,
           const parsed_uri& url, hs_scratch_t* scratch)
{
    // This buffer is used for reading and must be persisted
    beast::flat_buffer buffer;

#if 0
            fmt::print(FMT_STRING("Failed to parse \"{}\" as an URI address.\n"), url);
#endif

    if (url.protocol() == "file")
    {
        fetch_file(context, url, scratch);
    }
    else
    {
        fetch_web(context, ctx, resolver, ioc, scratch, url, buffer);
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

    parsed_uri url;

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

    auto url = parse_url(start_url);

    if (!url)
    {
        // todo
    }

    context.enqueue(url.value());

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
template <typename T, typename Count>
void fetch(T&& url, Count&& count, double time_bound)
{
    using namespace boost::ut;
    auto result = execute(scn::make_view(url));
    expect(eq(result.pages_crawled, count));
    const auto elapsed_ms = result.timer.get_elapsed_ms();
    expect(elapsed_ms > 0._d);
    expect(le(elapsed_ms, time_bound));
    //boost::ut::log << elapsed_ns;
    fmt::print(FMT_STRING("> {} ms\n"), elapsed_ms);
};

void test_suite()
{
    using namespace boost::ut;
    using namespace std::literals;

    "online"_test = []
    {
        skip | "google.com"_test = []
        {
            fetch("https://google.com 4"s, 1, 1000. * 1000);
        };

        skip | "ya.ru"_test = []
        {
            fetch("https://ya.ru 6"s, 1, 1000. * 1000);
        };
    };

    "offline"_test = []
    {
        skip | "test_data(1)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 1"s, 500, 7. * 1000);
        };
        skip | "test_data(2)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 2"s, 500, 5. * 1000);
        };
        "test_data(4)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 4"s, 500, 2. * 1000);
        };
        "test_data(5)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 5"s, 500, 2. * 1000);
        };
        "test_data(6)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 6"s, 500, 2. * 1000);
        };
        "test_data(7)"_test = []
        {
            fetch("file://C:/Users/Andrew/Desktop/test_data/0.html 7"s, 500, 2. * 1000);
        };
    };
}
#pragma warning( pop )
