﻿#include <fstream>
#include <sstream>
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

#define NOGDI

#include "peglib.h"
#include "debug_assert.hpp"
#include "ut.hpp"

// cppppack: embed point

struct name_lookup_context;

struct assert_module : debug_assert::default_handler, debug_assert::set_level<1>
{
};

std::string slurp(std::ifstream & in) {
    std::stringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

constexpr char grammar[] =
    R"(
expression <- _ (val / var / add / if / let / function / call) _
val <- '(' 'val' integer ')'
var <- '(' 'var' ident ')'
add <- '(' 'add' expression expression ')'
if <- '(' 'if' expression expression 'then' expression 'else' expression ')'
let <- '(' 'let' ident '=' expression 'in' expression ')'
function <- '(' 'function' ident expression ')'
call <- '(' 'call' expression expression ')'

ident      <- _ < [a-zA-Z] [a-zA-Z0-9]* > _
integer    <- sign number
sign       <- _ < [-+]? > _
number     <- _ < [0-9]+ > _
~_         <- [ \t\r\n]*
~__        <- ![a-zA-Z0-9_] _
    )";

enum class condition_operator_kind
{
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal
};

// For convenient std::visit (+deduction guide)
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...)->overloaded<Ts...>;

struct value;
struct integer_value;
struct function_value;
struct generator_value;
struct sequence_value;
struct sequence_iterator_value;
struct condition_value;
struct call_value;
struct variable;
struct variable_late_binding;

struct sequence_iterator_end_t
{
    explicit sequence_iterator_end_t() = default;
};

const sequence_iterator_end_t sequence_iterator_end;

std::shared_ptr<value> next(std::shared_ptr<value> source);
void bind(std::shared_ptr<value> expression, std::shared_ptr<value> source);

struct value
{
    virtual ~value() = default;

    virtual std::shared_ptr<value> evaluate() = 0;

    virtual std::optional<std::partial_ordering> compare(const value* other) const
    {
        return std::nullopt;
    }

    static decltype(auto) compare(const value* const lhs, const value* const rhs)
    {
        auto result = lhs->compare(rhs);
        if (result.has_value())
        {
            return result;
        }

        result = rhs->compare(lhs);
        if (result.has_value())
        {
            return result;
        }

        return result;
    }

    static bool equal(const value* const lhs, const value* const rhs)
    {
        const auto result = compare(lhs, rhs);
        return std::is_eq(result.value_or(std::partial_ordering::unordered));
    }

    friend bool operator==(const value& lhs, const value& rhs)
    {
        return equal(&lhs, &rhs);
    }

    friend bool operator!=(const value& lhs, const value& rhs)
    {
        return !(lhs == rhs);
    }

    friend std::partial_ordering operator<=>(const value& lhs, const value& rhs)
    {
        auto result = compare(&lhs, &rhs);
        if (result.has_value())
        {
            return result.value();
        }

        return std::partial_ordering::unordered;
    }

    template <typename T, typename... Args, std::enable_if_t<std::is_base_of_v<value, T>>* = nullptr>
    static decltype(auto) create(Args&&... params)
    {
        // ReSharper disable once CppSmartPointerVsMakeFunction
        return std::shared_ptr<T>(::new T(std::forward<Args>(params)...));
    }
};

namespace operations
{
    template <typename T1, typename T2>
    struct op
    {
    };

    template <typename T>
    struct op_with_equality_tag
    {
    };

    template <typename T>
    struct op_without_equality_tag
    {
    };

    template <typename T>
    struct op_with_comparison_tag
    {
    };

    template <typename T>
    struct op_without_comparison_tag
    {
    };

    template <typename T1, typename T2, typename Derived>
    struct op_with_equality : op<T1, T2>, op_with_equality_tag<std::add_const_t<T1>>, op_with_equality_tag<std::add_const_t<T2>>
    {
        static_assert(std::is_base_of_v<value, T1>);
        using t1 = std::shared_ptr<std::add_const_t<T1>>;
        static_assert(std::is_base_of_v<value, T2>);
        using t2 = std::shared_ptr<std::add_const_t<T2>>;

        bool not_equal(t1 lhs, t2 rhs)
        {
            return !Derived::equal(std::move(lhs), std::move(rhs));
        }
    };

    template <typename T1, typename T2>
    struct op_without_equality : op<T1, T2>, op_without_equality_tag<std::add_const_t<T1>>, op_without_equality_tag<std::add_const_t<T2>>
    {
        static_assert(std::is_base_of_v<value, T1>);
        using t1 = std::shared_ptr<std::add_const_t<T1>>;
        static_assert(std::is_base_of_v<value, T2>);
        using t2 = std::shared_ptr<std::add_const_t<T2>>;

        bool not_equal(const t1, const t2)
        {
            return false;
        }
    };

    template <typename T1, typename T2, typename Derived>
    struct op_with_comparison : op<T1, T2>, op_with_comparison_tag<std::add_const_t<T1>>, op_with_comparison_tag<std::add_const_t<T2>>
    {
        static_assert(std::is_base_of_v<value, T1>);
        using t1 = std::shared_ptr<std::add_const_t<T1>>;
        static_assert(std::is_base_of_v<value, T2>);
        using t2 = std::shared_ptr<std::add_const_t<T2>>;

        bool greater_equal(t1 lhs, t2 rhs)
        {
            return !Derived::less(std::move(lhs), std::move(rhs));
        }

        bool greater(t1 lhs, t2 rhs)
        {
            return Derived::less(std::move(rhs), std::move(lhs));
        }

        bool less_equal(t1 lhs, t2 rhs)
        {
            return !Derived::greater(std::move(lhs), std::move(rhs));
        }
    };

    template <typename T1, typename T2>
    struct op_without_comparison : op<T1, T2>, op_without_comparison_tag<std::add_const_t<T1>>, op_without_comparison_tag<std::add_const_t<T2>>
    {
        static_assert(std::is_base_of_v<value, T1>);
        using t1 = std::shared_ptr<std::add_const_t<T1>>;
        static_assert(std::is_base_of_v<value, T2>);
        using t2 = std::shared_ptr<std::add_const_t<T2>>;

        bool greater_equal(const t1, const t2)
        {
            return false;
        }

        bool greater(const t1, const t2)
        {
            return false;
        }

        bool less_equal(const t1, const t2)
        {
            return false;
        }
    };

    template <typename Derived>
    struct predicate
    {
        template <typename T1, typename T2, typename C1 = std::add_const_t<T1>, typename C2 = std::add_const_t<T2>, typename P1 = std::shared_ptr<T1>, typename P2 = std::shared_ptr<T2>>
        bool greater_equal(P1 lhs, P2 rhs)
        {
            auto lhs_c1 = std::const_pointer_cast<C1>(lhs);
            auto lhs_c2 = std::dynamic_pointer_cast<C2>(lhs_c1);
            auto rhs_c2 = std::const_pointer_cast<C2>(rhs);
            auto rhs_c1 = std::dynamic_pointer_cast<C1>(rhs_c2);
            auto result = false;
            if constexpr (std::is_base_of_v<op_with_comparison<T1, T2, Derived>, Derived>)
            {
                result = result || Derived::greater_equal(std::move(lhs), std::move(rhs));
            }
            if constexpr (std::is_base_of_v<op_with_comparison<T1, C2, Derived>, Derived>)
            {
                result = result || Derived::greater_equal(std::move(lhs), std::move(rhs_c2));
            }
            if constexpr (std::is_base_of_v<op_with_comparison<C1, T2, Derived>, Derived>)
            {
                result = result || Derived::greater_equal(std::move(lhs_c1), std::move(rhs));
            }
            if constexpr (std::is_base_of_v<op_with_comparison<C1, C2, Derived>, Derived>)
            {
                result = result || Derived::greater_equal(std::move(lhs_c1), std::move(rhs_c2));
            }
            if (!result && lhs_c1)
            {

            }
            if constexpr (std::is_base_of_v<op_with_comparison_tag<C1>, Derived> && std::is_base_of_v<op_with_comparison_tag<C2>, Derived>)
            {
                if constexpr (std::is_base_of_v<op_with_comparison<C1, C2, Derived>, Derived>)
                {
                    result = result || Derived::greater_equal(std::move(lhs_c1), std::move(rhs));
                }
                if constexpr (std::is_base_of_v<op_with_comparison<C1, C2, Derived>, Derived>)
                {
                    result = result || Derived::greater_equal(std::move(lhs), std::move(rhs_c2));
                }
                if constexpr (std::is_base_of_v<op_with_comparison<C1, C2, Derived>, Derived>)
                {
                    result = result || Derived::greater_equal(std::move(lhs_c1), std::move(rhs_c2));
                }
            }
            return result;
        }

        bool greater(const t1, const t2)
        {
            return false;
        }

        bool less_equal(const t1, const t2)
        {
            return false;
        }
    };
}

bool operator==(const std::shared_ptr<const expression>& lhs, const std::shared_ptr<const expression>& rhs)
{
    if (!lhs && !rhs)
        return true;

    if (lhs && !rhs || !lhs && rhs)
        return false;

    return *lhs == *rhs;
}

bool operator!=(const std::shared_ptr<const expression>& lhs, const std::shared_ptr<const expression>& rhs)
{
    return !(lhs == rhs);
}

bool operator==(const std::shared_ptr<expression>& lhs, const std::shared_ptr<expression>& rhs)
{
    return std::const_pointer_cast<const expression>(lhs) == std::const_pointer_cast<const expression>(rhs);
}

bool operator!=(const std::shared_ptr<expression>& lhs, const std::shared_ptr<expression>& rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const expression& lhs, const expression& rhs)
{
    return !(lhs == rhs);
}

struct variable_late_binding final
{
    std::string name;

    friend bool operator==(const variable_late_binding& lhs, const variable_late_binding& rhs)
    {
        return lhs.name == rhs.name;
    }

    friend bool operator!=(const variable_late_binding& lhs, const variable_late_binding& rhs)
    {
        return !(lhs == rhs);
    }

    std::shared_ptr<variable> bound_variable;

    friend std::size_t hash_value(const variable_late_binding& obj) noexcept
    {
        std::size_t seed = 0x32ECF885;
        seed ^= (seed << 6) + (seed >> 2) + 0x161D3D4A + std::hash<decltype(name)>()(obj.name);
        return seed;
    }

    template <typename... Args>
    static decltype(auto) create(Args&&... params)
    {
        // ReSharper disable once CppSmartPointerVsMakeFunction
        return std::shared_ptr<variable_late_binding>(::new variable_late_binding(std::forward<Args>(params)...));
    }

    variable_late_binding& bind(const std::shared_ptr<name_lookup_context>& context);

private:
    explicit variable_late_binding(std::string name)
        : name(std::move(name))
    {
    }

    variable_late_binding(std::string name, std::shared_ptr<variable> bound_variable)
        : name(std::move(name)),
          bound_variable(std::move(bound_variable))
    {
    }
};

namespace std
{
    template<> struct hash<variable_late_binding>
    {
        std::size_t operator()(variable_late_binding const& obj) const noexcept
        {
            return hash_value(obj);
        }
    };
}

struct integer_value final : value, std::enable_shared_from_this<integer_value>
{
    int64_t value;

    ~integer_value() override = default;

    integer_value(const integer_value& other) = default;
    integer_value(integer_value&& other) noexcept = default;
    integer_value& operator=(const integer_value& other) = default;
    integer_value& operator=(integer_value&& other) noexcept = default;

    std::shared_ptr<::value> evaluate() override
    {
        return shared_from_this();
    }

private:
    explicit integer_value(const int64_t value)
        : value(value)
    {
    }

    friend struct value;
};

struct function_value : value
{
    std::shared_ptr<name_lookup_context> scope;

    std::shared_ptr<value> execute(std::shared_ptr<sequence_value> arguments)
    {
        return execute_impl(std::move(arguments));
    }

protected:
    virtual std::shared_ptr<value> execute_impl(std::shared_ptr<sequence_value> arguments) = 0;

    explicit function_value(std::shared_ptr<name_lookup_context> scope)
        : scope(std::move(scope))
    {
    }
};

struct native_function_value : function_value
{
protected:
    explicit native_function_value(std::shared_ptr<name_lookup_context> scope)
        : function_value(std::move(scope))
    {
    }
};

struct add_function_value final : native_function_value
{
    std::shared_ptr<value> execute_impl(std::shared_ptr<sequence_value> arguments) override;

private:
    explicit add_function_value(std::shared_ptr<name_lookup_context> scope)
        : native_function_value(std::move(scope))
    {
    }

    friend struct value;
};

struct managed_function_value final : function_value
{
    std::shared_ptr<value> argument; // variable / tuple[sequence]
    std::shared_ptr<value> body;

    std::shared_ptr<value> execute_impl(std::shared_ptr<sequence_value> arguments) override;

private:
    managed_function_value(std::shared_ptr<value> argument, std::shared_ptr<name_lookup_context> scope,
                           std::shared_ptr<value> value)
        : function_value(std::move(scope)),
          argument(std::move(argument)),
          body(std::move(value))
    {
    }

    friend struct value;
};

struct generator_value final : value
{
    // f(x) for x in args                                  => x
    // f(x) for x in args if predicate(x)                  => x
    // f(x) for [i, x] in enumerate(args) if predicate(x)  => x
    std::shared_ptr<value> expression;
    // f(x) for x in args                                  => args
    // f(x) for x in args if predicate(x)                  => args
    // f(x) for [i, x] in enumerate(args) if predicate(x)  => enumerate(args)
    std::shared_ptr<value> source;
    // f(x) for x in args                                  => f(x)
    // f(x) for x in args if predicate(x)                  => [predicate(x) ? f(x) : <skip>]
    // f(x) for x in args if predicate(x) else g(x)        => [predicate(x) ? f(x) : g(x)]
    std::variant<std::shared_ptr<condition_value>, std::shared_ptr<value>> transforms;

    std::shared_ptr<value> next();

private:
    generator_value(std::shared_ptr<value> expression, std::shared_ptr<value> source,
                    std::shared_ptr<condition_value> transforms)
        : expression(std::move(expression)),
          source(std::move(source)),
          transforms(std::move(transforms))
    {
    }

    generator_value(std::shared_ptr<value> expression, std::shared_ptr<value> source,
                    std::shared_ptr<value> transforms)
        : expression(std::move(expression)),
          source(std::move(source)),
          transforms(std::move(transforms))
    {
    }

    friend struct value;
};

struct sequence_value final : value, std::enable_shared_from_this<sequence_value>
{
    using value_type = std::shared_ptr<value>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using const_iterator = sequence_iterator_value;
    using iterator = const_iterator;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    std::deque<value_type> values;
    std::shared_ptr<generator_value> generator;

    const_iterator begin();
    const_iterator end();

    const_iterator cbegin();
    const_iterator cend();

    size_type size();

    std::shared_ptr<value> evaluate() override;

private:
    explicit sequence_value(std::shared_ptr<generator_value> generator_value)
        : generator(std::move(generator_value))
    {
    }

    explicit sequence_value(std::deque<value_type> values)
        : values(std::move(values))
    {
    }

    explicit sequence_value() = default;

    friend struct value;
};

struct sequence_iterator_value final : value, std::enable_shared_from_this<sequence_iterator_value>
{
    using value_type = std::shared_ptr<value>;
    using reference = value_type&;
    using pointer = value_type*;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    std::shared_ptr<sequence_value> sequence;
    std::size_t index;
    bool end;

    decltype(auto) operator*()
    {
        DEBUG_ASSERT(!end, assert_module{});
        return get();
    }

    decltype(auto) operator->()
    {
        DEBUG_ASSERT(!end, assert_module{});
        return get().operator->();
    }

    decltype(auto) operator++()
    {
        DEBUG_ASSERT(!end, assert_module{});
        ++index;
        return *this;
    }

    decltype(auto) operator--()
    {
        DEBUG_ASSERT(!end, assert_module{});
        --index;
        return *this;
    }

    decltype(auto) operator++(int)
    {
        auto ip = *this;
        ++*this;
        return ip;
    }

    decltype(auto) operator--(int)
    {
        auto ip = *this;
        --*this;
        return ip;
    }

    friend bool operator==(const sequence_iterator_value& lhs, const sequence_iterator_value& rhs)
    {
        return lhs.sequence == rhs.sequence
            && (lhs.index == rhs.index || (lhs.end && rhs.end));
    }

    friend bool operator!=(const sequence_iterator_value& lhs, const sequence_iterator_value& rhs)
    {
        return !(lhs == rhs);
    }

    std::shared_ptr<value> evaluate() override
    {
        return shared_from_this();
    }

private:
    explicit sequence_iterator_value(std::shared_ptr<sequence_value> sequence_value)
        : sequence(std::move(sequence_value)), index(0u), end(false)
    {
    }
    explicit sequence_iterator_value(std::shared_ptr<sequence_value> sequence_value, sequence_iterator_end_t)
        : sequence(std::move(sequence_value)), index(std::numeric_limits<decltype(index)>::max()), end(true)
    {
    }

    void generate_next() const
    {
        DEBUG_ASSERT(!end, assert_module{});

        auto&& generator = sequence->generator;
        if (generator)
        {
            sequence->values.push_back(generator->next());
        }
    }

    [[nodiscard]] std::shared_ptr<value> get()
    {
        auto&& values = sequence->values;
        if (index >= values.size())
        {
            generate_next();
        }

        if (index >= values.size())
        {
            end = true;
            return nullptr;
        }

        DEBUG_ASSERT(!end, assert_module{});

        return values[index];
    }

    friend struct value;
    friend struct sequence_value;
};

struct call_value final : value
{
    std::shared_ptr<variable_late_binding> what;
    std::shared_ptr<sequence_value> arguments;

    std::shared_ptr<value> evaluate() override;

private:
    call_value(variable_late_binding what, std::shared_ptr<sequence_value> sequence_value)
        : what(std::move(what)),
          arguments(std::move(sequence_value))
    {
    }

    friend struct value;
};

struct condition_value final : value
{
    std::shared_ptr<value> left;
    condition_operator_kind kind;
    std::shared_ptr<value> right;
    std::shared_ptr<value> suite_true;
    std::shared_ptr<value> suite_false;

    bool evaluate_condition();

    std::shared_ptr<value> evaluate() override
    {
        return evaluate_condition() ? suite_true : suite_false;
    }

private:
    condition_value(std::shared_ptr<value> left, const condition_operator_kind kind, std::shared_ptr<value> right,
        std::shared_ptr<value> suite_true, std::shared_ptr<value> suite_false)
        : left(std::move(left)),
          kind(kind),
          right(std::move(right)),
          suite_true(std::move(suite_true)),
          suite_false(std::move(suite_false))
    {
    }

    friend struct value;
};

struct variable final : value
{
    std::string name;
    std::shared_ptr<value> value;

    std::shared_ptr<::value> evaluate() override
    {
        return value->evaluate();
    }

private:
    explicit variable(std::string name)
        : name(std::move(name))
    {
    }

    friend struct value;
};

namespace operations
{
    template <>
    struct op<integer_value> : op_with_equality<integer_value, op<integer_value>>,
                               op_with_comparison<integer_value, op<integer_value>>
    {
        bool equal(const std::shared_ptr<const integer_value>& lhs, const std::shared_ptr<const integer_value>& rhs)
        {
            return lhs->value == rhs->value;
        }

        bool less(const std::shared_ptr<const integer_value>& lhs, const std::shared_ptr<const integer_value>& rhs)
        {
            return lhs->value < rhs->value;
        }
    };
}

void test_suite();

struct context_base
{
    template <typename T, typename... Args, std::enable_if_t<std::is_base_of_v<context_base, T>> * = nullptr>
    static decltype(auto) create(Args&&... params)
    {
        // ReSharper disable once CppSmartPointerVsMakeFunction
        return std::shared_ptr<T>(::new T(std::forward<Args>(params)...));
    }
};

struct name_lookup_context final : context_base, std::enable_shared_from_this<name_lookup_context>
{
    std::shared_ptr<name_lookup_context> parent;
    std::unordered_map<std::string, std::shared_ptr<variable>> defined;

    std::shared_ptr<variable_late_binding> operator[](const std::string& key)
    {
        const auto local_cache = bound_cache_.find(key);

        if (local_cache != bound_cache_.end())
        {
            return local_cache->second;
        }

        auto local_cache_binding = variable_late_binding::create(key);
        bound_cache_.emplace(key, local_cache_binding);
        return local_cache_binding;
    }

    std::shared_ptr<variable> define(const std::string& key)
    {
        return defined.emplace(key, value::create<variable>(key)).first->second;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<variable_late_binding>> bound_cache_;

    explicit name_lookup_context(std::shared_ptr<name_lookup_context> parent)
        : parent(std::move(parent))
    {
    }

    explicit name_lookup_context() = default;

    std::shared_ptr<variable> find_definition(const std::string& key)
    {
        const auto local_cache = bound_cache_.find(key);
        std::shared_ptr<variable_late_binding> local_cache_binding;

        if (local_cache != bound_cache_.end())
        {
            local_cache_binding = local_cache->second;

            if (local_cache_binding->bound_variable)
            {
                return local_cache_binding->bound_variable;
            }
        }
        else
        {
            local_cache_binding = variable_late_binding::create(key);
            bound_cache_.emplace(key, local_cache_binding);
        }

        const auto local_definition = defined.find(key);
        if (local_definition != defined.end())
        {
            local_cache_binding->bound_variable = local_definition->second;
            return local_cache_binding->bound_variable;
        }

        local_cache_binding->bound_variable = parent ? parent->find_definition(key) : nullptr;
        return local_cache_binding->bound_variable;
    }


    friend struct context_base;
    friend struct variable_late_binding;
};

variable_late_binding& variable_late_binding::bind(const std::shared_ptr<name_lookup_context>& context)
{
    if (!bound_variable)
    {
        const auto variable = context->find_definition(name);
        if (bound_variable != variable)
            bound_variable = variable;
    }

    return *this;
}

std::shared_ptr<value> add_function_value::execute_impl(std::shared_ptr<sequence_value> arguments)
{
    int64_t result = 0;
    for (auto&& value : *arguments)
    {
        if (const auto integral = std::dynamic_pointer_cast<integer_value>(value))
        {
            result += integral->value;
        }
        else
        {
            throw std::exception("add: expected integer arguments");
        }
    }

    return create<integer_value>(result);
}

std::shared_ptr<value> managed_function_value::execute_impl(std::shared_ptr<sequence_value> arguments)
{
    ::bind(argument, std::static_pointer_cast<value>(arguments));
    return body;
}

std::shared_ptr<value> generator_value::next()
{
    DEBUG_ASSERT(source, assert_module{});

    while (true)
    {
        auto item = ::next(source);
        if (!item)
        {
            return item;
        }

        bind(expression, item);

        auto transformed = ::std::visit(
            [](auto&& arg) { return arg->evaluate(); },
            transforms
        );

        if (!transformed)
        {
            continue;
        }

        return transformed;
    }
}

static_assert(std::is_same_v<std::iterator_traits<sequence_value::const_iterator>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<std::iterator_traits<sequence_value::const_iterator>::iterator_category, std::bidirectional_iterator_tag>);

sequence_value::const_iterator sequence_value::begin()
{
    return cbegin();
}

sequence_value::const_iterator sequence_value::end()
{
    return cend();
}

sequence_value::const_iterator sequence_value::cbegin()
{
    return const_iterator(shared_from_this());
}

sequence_value::const_iterator sequence_value::cend()
{
    return const_iterator(shared_from_this(), sequence_iterator_end);
}

sequence_value::size_type sequence_value::size()
{
    return std::distance(cbegin(), cend());
}

std::shared_ptr<value> sequence_value::evaluate()
{
    decltype(values) result;

    for (auto&& item : *this)
    {
        result.push_back(item->evaluate());
    }

    return create<sequence_value>(result);
}

struct ast_tree_context final : context_base
{
    std::shared_ptr<ast_tree_context> parent;
    std::shared_ptr<name_lookup_context> name_lookup;

private:
    explicit ast_tree_context(std::shared_ptr<ast_tree_context> ast_tree_context)
        : parent(std::move(ast_tree_context)), name_lookup(create<name_lookup_context>(parent->name_lookup))
    {
    }

    explicit ast_tree_context() : name_lookup(create<name_lookup_context>())
    {
    }

    friend struct context_base;
};

std::shared_ptr<value> eval(const std::shared_ptr<ast_tree_context>& ast_context, const std::shared_ptr<peg::Ast>& ast)
{
    auto&& node_name = ast->name;
    if (node_name == "integer")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "sign", assert_module{});
        DEBUG_ASSERT(nodes[1]->name == "number", assert_module{});
        auto&& sign_token = nodes[0]->token;
        auto&& number_token = nodes[1]->token;

        const int64_t sign = sign_token == "-" ? -1 : 1;

        int64_t result;
        if (auto [p, ec] = std::from_chars(number_token.data(), number_token.data() + number_token.size(), result);
            ec == std::errc())
        {
            return value::create<integer_value>(sign * result);
        }

        throw std::exception("Value cannot be parsed as integer");
    }

    if (node_name == "let")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 3, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "ident", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[2]->original_name == "expression", assert_module{});
        auto&& identifier = nodes[0]->token;

        const auto context = context_base::create<ast_tree_context>(ast_context);
        const auto variable = context->name_lookup->define(identifier);

        variable->value = eval(context, nodes[1]);
        return eval(context, nodes[2]);
    }

    if (node_name == "var")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "ident", assert_module{});
        auto&& identifier = nodes[0]->token;

        return (*ast_context->name_lookup)[identifier]->bind(ast_context->name_lookup).bound_variable;
    }

    if (node_name == "val")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "integer", assert_module{});
        return eval(ast_context, nodes[0]);
    }

    DEBUG_UNREACHABLE(assert_module{});

    return nullptr;
#if 0
    auto&& nodes = ast->nodes;
    auto result = eval(nodes[0]);
    for (auto i = 1u; i < nodes.size(); i += 2) {
        auto num = eval(nodes[i + 1]);
        auto ope = nodes[i]->token[0];
        switch (ope) {
        case '+': result += num; break;
        case '-': result -= num; break;
        case '*': result *= num; break;
        case '/': result /= num; break;
        }
    }

    return result;
#endif
};

std::shared_ptr<value> peg_parser(const std::string& source)
{
    peg::parser parser;

    DEBUG_ASSERT(parser.load_grammar(grammar), assert_module{});

    parser.enable_ast();

    std::shared_ptr<peg::Ast> ast;
    if (!parser.parse(source.c_str(), ast))
        return nullptr;

    ast = peg::AstOptimizer(true).optimize(ast);

    const auto context = context_base::create<ast_tree_context>();
    return eval(context, ast);
}

int main() // NOLINT(bugprone-exception-escape)
{
    std::ifstream fin("input.txt");

    if (!fin.is_open())
    {
        test_suite();
        return 0;
    }

    const auto source = slurp(fin);

    std::ofstream fout("output.txt");

    auto ast = peg_parser(source);

    fout << "(val 0)" << std::endl;

    return 0;
}

std::shared_ptr<value> call_value::evaluate()
{
    const auto function = std::dynamic_pointer_cast<function_value>(what->bound_variable);
    if (!function)
    {
        throw std::exception("Can only call functions");
    }

    const auto arguments = std::dynamic_pointer_cast<sequence_value>(this->arguments->evaluate());

    if (!arguments)
    {
        throw std::exception("Can only use sequences as arguments");
    }

    return function->execute(arguments);
}

bool condition_value::evaluate_condition()
{
    auto left = this->left->evaluate();
    auto right = this->right->evaluate();
    switch (this->kind) {
        case condition_operator_kind::equal:
            return left == right;
        case condition_operator_kind::not_equal: break;
        case condition_operator_kind::less: break;
        case condition_operator_kind::less_equal: break;
        case condition_operator_kind::greater: break;
        case condition_operator_kind::greater_equal: break;
        default: ;
    }
}

void test_suite()
{
    using namespace boost::ut;

    "examples"_test = [] {
        "1"_test = [] {
            auto ast = peg_parser(R"((let K = (val 10) in
                                              (add
                                                  (val 5)
                                                  (var K))))");

            expect(false == true_b);
        };
        "2"_test = [] {
            auto ast = peg_parser(R"((let A = (val 20) in
                                              (let B = (val 30) in
                                                  (if
                                                      (var A)
                                                      (add
                                                          (var B)
                                                          (val 3)
                                                      )
                                                      then
                                                      (val 10)
                                                      else
                                                      (add
                                                          (var B)
                                                          (val 1)
                                                      )
                                                  )
                                              )
                                           ))");

            expect(false == true_b);
        };
        "3"_test = [] {
            auto ast = peg_parser(R"((let F = (function arg (add (var arg) (val 1))) in
                                              (let V = (val -1) in
                                                  (call (var F) (var V)))))");

            expect(false == true_b);
        };
        "4"_test = [] {
            auto ast = peg_parser(R"((add (var A) (var B)))");

            expect(false == true_b);
        };
    };
}