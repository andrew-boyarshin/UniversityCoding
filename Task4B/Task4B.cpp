#include <fstream>
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
#include <string_view>
#include <string>

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

constexpr char grammar[] =
    R"(

expression <- _ (val / var / if / let / function / call / free_call) _
val        <- '(' 'val' integer ')'
var        <- '(' 'var' ident ')'
if         <- '(' 'if' expression expression 'then' expression 'else' expression ')'
let        <- '(' 'let' ident '=' expression 'in' expression ')'
function   <- '(' 'function' bind expression ')'
call       <- '(' 'call' expression expression ')'
free_call  <- '(' ident expression* ')'

bind       <- (tuple / ident)
ident      <- _ < [a-zA-Z] [a-zA-Z0-9]* > _
integer    <- sign number
sign       <- _ < [-+]? > _
number     <- _ < [0-9]+ > _
tuple      <- _ '[' bind (',' bind)* ']' _
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

class format_context;

struct value;
struct integer_value;
struct function_value;
struct generator_value;
struct sequence_value;

template <bool Const>
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

struct format_raise_precedence final
{
    std::int16_t difference;

    explicit format_raise_precedence(const std::int16_t difference)
        : difference(difference)
    {
    }

    ~format_raise_precedence() = default;

    format_raise_precedence(const format_raise_precedence& other) = default;
    format_raise_precedence(format_raise_precedence&& other) noexcept = default;
    format_raise_precedence& operator=(const format_raise_precedence& other) = default;
    format_raise_precedence& operator=(format_raise_precedence&& other) noexcept = default;
};

struct format_set_precedence final
{
    std::int16_t value;

    explicit format_set_precedence(const std::int16_t value)
        : value(value)
    {
    }

    ~format_set_precedence() = default;

    format_set_precedence(const format_set_precedence& other) = default;
    format_set_precedence(format_set_precedence&& other) noexcept = default;
    format_set_precedence& operator=(const format_set_precedence& other) = default;
    format_set_precedence& operator=(format_set_precedence&& other) noexcept = default;
};

std::shared_ptr<value> next(std::shared_ptr<value> source);

template <typename T>
std::optional<std::size_t> len(T&& source);

void bind(std::shared_ptr<value> expression, std::shared_ptr<value> source);

struct value
{
    virtual ~value() = default;

    virtual std::shared_ptr<value> evaluate() = 0;
    virtual format_context& format(format_context& output) const = 0;

protected:
    virtual std::optional<std::partial_ordering> compare(const value*) const
    {
        return std::nullopt;
    }

private:
    static decltype(auto) compare_impl(const value* const lhs, const value* const rhs)
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

public:
    static decltype(auto) compare(const value* const lhs, const value* const rhs)
    {
        const auto result = compare_impl(lhs, rhs);
        return result.value_or(std::partial_ordering::unordered);
    }

    friend std::partial_ordering operator<=>(const value& lhs, const value& rhs)
    {
        return compare(&lhs, &rhs);
    }

    template <typename T, typename... Args, std::enable_if_t<std::is_base_of_v<value, T>>* = nullptr>
    static decltype(auto) create(Args&&... params)
    {
        // ReSharper disable once CppSmartPointerVsMakeFunction
        return std::shared_ptr<T>(::new T(std::forward<Args>(params)...));
    }
};

class format_context
{
    std::ostream* stream_;
    bool force_explicit_call_;

public:
    bool force_explicit_call() const
    {
        return force_explicit_call_;
    }

    void set_force_explicit_call(const bool force_explicit_call)
    {
        force_explicit_call_ = force_explicit_call;
    }

    class scope;

private:
    scope* current_scope_ = nullptr;

    void raise_precedence_impl(const format_raise_precedence& value)
    {
        const auto scope = current_scope();
        DEBUG_ASSERT(scope, assert_module{});
        auto precedence = scope->precedence();
        DEBUG_ASSERT(precedence.has_value(), assert_module{});
        const auto new_precedence = precedence.value() - value.difference;
        set_precedence_impl(format_set_precedence(new_precedence));
    }

    void set_precedence_impl(const format_set_precedence& value)
    {
        const auto scope = current_scope();
        DEBUG_ASSERT(scope, assert_module{});
        scope->set_precedence(value.value);
    }

public:
    explicit format_context(std::ostream& stream)
        : stream_(&stream)
    {
    }

    template<typename T>
    format_context& operator<<(T&& obj)
    {
        using obj_t = decltype(obj);
        using obj_decay_t = type_decay_t<obj_t>;
        if constexpr (std::is_base_of_v<value, obj_decay_t>)
        {
            static_assert(!std::is_pointer_v<obj_t>);
            if constexpr (is_smart_pointer_v<std::remove_cvref_t<obj_t>>)
            {
                return obj->format(*this);
            }
            else
            {
                static_assert(std::is_reference_v<obj_t>);
                return obj.format(*this);
            }
        }
        else if constexpr (std::is_same_v<format_raise_precedence, obj_decay_t>)
        {
            raise_precedence_impl(obj);
        }
        else if constexpr (std::is_same_v<format_set_precedence, obj_decay_t>)
        {
            set_precedence_impl(obj);
        }
        else if constexpr (std::is_same_v<format_context, obj_decay_t>)
        {
            DEBUG_ASSERT(std::addressof(obj) == this, assert_module{});
        }
        else
        {
            *stream_ << obj;
        }

        return *this;
    }

    [[nodiscard]] scope* current_scope() const
    {
        return current_scope_;
    }

    [[nodiscard]] bool parameter_pack() const
    {
        const auto scope = current_scope();
        return scope ? scope->parameter_pack() : false;
    }

    class scope final
    {
        format_context* const context_;
        scope* const parent_;
        std::optional<std::int16_t> precedence_;
        bool wrap_in_brackets_ = false;
        std::optional<bool> parameter_pack_;

        [[nodiscard]] bool is_lower_precedence(const std::int16_t other) const
        {
            const auto precedence = this->precedence();
            return precedence.has_value() ? precedence.value() < other : false;
        }

    public:
        scope(format_context& context, std::int16_t precedence) : scope(context)
        {
            precedence_ = precedence;
            wrap_in_brackets_ = parent_ ? parent_->is_lower_precedence(precedence) : false;

            if (wrap_in_brackets_)
            {
                context << '(';
            }
        }

        explicit scope(format_context& context) : context_(&context), parent_(context.current_scope())
        {
            context.current_scope_ = this;
        }

        ~scope()
        {
            if (wrap_in_brackets_)
            {
                try
                {
                    *context_ << ')';
                }
                catch (...)
                {
                    // Yay, possible I/O exception doesn't escape destructor!
                }
            }

            context_->current_scope_ = parent_;
        }

        [[nodiscard]] scope* parent() const
        {
            return parent_;
        }

        [[nodiscard]] std::optional<std::int16_t> precedence() const
        {
            if (precedence_.has_value())
            {
                return precedence_;
            }

            auto* parent = parent_;
            while (parent)
            {
                if (parent->precedence_.has_value())
                {
                    return parent->precedence_.value();
                }

                parent = parent->parent_;
            }

            return std::nullopt;
        }

        void set_precedence(const std::int16_t precedence)
        {
            precedence_ = precedence;
        }

        [[nodiscard]] bool wrap_in_brackets() const
        {
            return wrap_in_brackets_;
        }

        [[nodiscard]] bool parameter_pack() const
        {
            if (parameter_pack_.has_value())
            {
                return parameter_pack_.value();
            }

            auto* parent = parent_;
            while (parent)
            {
                if (parent->parameter_pack_.has_value())
                {
                    return parent->parameter_pack_.value();
                }

                parent = parent->parent_;
            }

            return false;
        }

        void set_parameter_pack(const bool parameter_pack)
        {
            parameter_pack_ = parameter_pack;
        }

        scope(const scope& other) = delete;
        scope(scope&& other) noexcept = delete;
        scope& operator=(const scope& other) = delete;
        scope& operator=(scope&& other) noexcept = delete;
    };
};

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

    explicit variable_late_binding(std::shared_ptr<variable> bound_variable);
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

    format_context& format(format_context& output) const override
    {
        return output << "(val " << value << ")";
    }

    std::shared_ptr<::value> evaluate() override
    {
        return shared_from_this();
    }

    std::optional<std::partial_ordering> compare(const ::value* other) const override
    {
        if (auto* other_as_int = dynamic_cast<const integer_value*>(other))
        {
            return value <=> other_as_int->value;
        }

        throw std::exception("Can only compare integer with integer");
    }

private:
    explicit integer_value(const int64_t value)
        : value(value)
    {
    }

    friend struct value;
};

struct function_value : value, std::enable_shared_from_this<function_value>
{
    std::shared_ptr<name_lookup_context> scope;

    std::shared_ptr<value> evaluate() override
    {
        return shared_from_this();
    }

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

    format_context& format(format_context& output) const override
    {
        throw std::exception("Shouldn't be formatted");
    }

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

    format_context& format(format_context& output) const override
    {
        output << "(function ";
        {
            DEBUG_ASSERT(!output.parameter_pack(), assert_module{});
            format_context::scope scope(output);
            scope.set_parameter_pack(true);
            output << argument;
        }
        return output << " " << body << ")";
    }

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
    using const_iterator = sequence_iterator_value<true>;
    using iterator = sequence_iterator_value<false>;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    mutable std::deque<value_type> values;
    std::shared_ptr<generator_value> generator;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    const_iterator cbegin() const;
    const_iterator cend() const;

    size_type size() const;

    std::shared_ptr<value> evaluate() override;

    void drain_generator() const;

    format_context& format(format_context& output) const override;

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

template <bool Const>
struct sequence_iterator_value final : value, std::enable_shared_from_this<sequence_iterator_value<Const>>
{
    using value_type_const = std::conditional_t<Const, const value, value>;
    using value_type = std::shared_ptr<value_type_const>;
    using reference = value_type&;
    using pointer = value_type*;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    std::variant<std::shared_ptr<const sequence_value>, std::shared_ptr<sequence_value>> sequence;
    std::size_t index;
    bool end;

    format_context& format(format_context& output) const override
    {
        DEBUG_ASSERT(!end, assert_module{});
        return get()->format(output);
    }

    decltype(auto) operator*() const
    {
        DEBUG_ASSERT(!end, assert_module{});
        return get();
    }

    decltype(auto) operator->() const
    {
        DEBUG_ASSERT(!end, assert_module{});
        return get().operator->();
    }

    decltype(auto) operator++()
    {
        DEBUG_ASSERT(!end, assert_module{});
        ++index;
        update_end_state();
        return *this;
    }

    decltype(auto) operator--()
    {
        DEBUG_ASSERT(!end, assert_module{});
        --index;
        update_end_state();
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

    std::shared_ptr<value> evaluate() override
    {
        return shared_from_this();
    }

private:
    explicit sequence_iterator_value(std::shared_ptr<sequence_value> sequence_value)
        : sequence(std::move(sequence_value)), index(0u), end(false)
    {
    }

    explicit sequence_iterator_value(std::shared_ptr<const sequence_value> sequence_value)
        : sequence(std::move(sequence_value)), index(0u), end(false)
    {
    }

    explicit sequence_iterator_value(std::shared_ptr<sequence_value> sequence_value, sequence_iterator_end_t)
        : sequence(std::move(sequence_value)), index(std::numeric_limits<decltype(index)>::max()), end(true)
    {
    }

    explicit sequence_iterator_value(std::shared_ptr<const sequence_value> sequence_value, sequence_iterator_end_t)
        : sequence(std::move(sequence_value)), index(std::numeric_limits<decltype(index)>::max()), end(true)
    {
    }

    void generate_next() const
    {
        DEBUG_ASSERT(!end, assert_module{});

        ::std::visit(
            overloaded
            {
                [](const std::shared_ptr<const sequence_value>& ptr)
                {
                    auto&& generator = ptr->generator;
                    if (generator)
                    {
                        throw std::exception("Const generation forbidden");
                    }
                },
                [](const std::shared_ptr<sequence_value>& ptr)
                {
                    auto&& generator = ptr->generator;
                    if (generator)
                    {
                        ptr->values.push_back(generator->next());
                    }
                }
            },
            sequence
        );
    }

    void update_end_state()
    {
        if (!end && index >= std::visit(
            [](auto&& ptr)
            {
                return ptr->values;
            },
            sequence
        ).size())
        {
            generate_next();
        }

        end = index >= std::visit(
            [](auto&& ptr)
            {
                return ptr->values;
            },
            sequence
        ).size();
    }

    [[nodiscard]] std::shared_ptr<value> get() const
    {
        DEBUG_ASSERT(!end, assert_module{});

        return
            std::visit(
                [](auto&& ptr)
                {
                    return ptr->values;
                },
                sequence
            )
            [index];
    }

    friend struct value;
    friend struct sequence_value;
};

template <bool Const1, bool Const2>
bool operator==(const sequence_iterator_value<Const1>& lhs, const sequence_iterator_value<Const2>& rhs)
{
    return lhs.sequence == rhs.sequence
        && (lhs.index == rhs.index || (lhs.end && rhs.end));
}

template <bool Const1, bool Const2>
bool operator!=(const sequence_iterator_value<Const1>& lhs, const sequence_iterator_value<Const2>& rhs)
{
    return !(lhs == rhs);
}

template <typename T>
decltype(auto) format_packed(format_context& output, T&& value)
{
    DEBUG_ASSERT(!output.parameter_pack(), assert_module{});
    format_context::scope scope(output);
    scope.set_parameter_pack(true);

    return output << std::forward<decltype(value)>(value);
}

struct call_value final : value
{
    std::shared_ptr<value> what;
    std::shared_ptr<sequence_value> arguments;

    std::shared_ptr<value> evaluate() override;

    format_context& format(format_context& output) const override
    {
        output << '(';

        if (output.force_explicit_call() && len(arguments) == 1)
            output << "call ";

        return output << format_packed(output, what) << ' ' << arguments << ')';
    }

private:
    call_value(std::shared_ptr<value> what, std::shared_ptr<sequence_value> sequence_value)
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

    bool evaluate_condition() const;

    std::shared_ptr<value> evaluate() override
    {
        return evaluate_condition() ? suite_true->evaluate() : suite_false->evaluate();
    }

    format_context& format(format_context& output) const override
    {
        return output << "(if " << left << " " << right << " " << suite_true << " " << suite_false << ")";
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

    format_context& format(format_context& output) const override
    {
        if (output.parameter_pack())
            return output << name;

        return output << "(var " << name << ")";
    }

private:
    explicit variable(std::string name)
        : name(std::move(name))
    {
    }

    friend struct value;
};

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

std::shared_ptr<value> next(const std::shared_ptr<value> source)
{
    if (auto it = std::dynamic_pointer_cast<sequence_iterator_value<true>>(source))
    {
        return ++*it, it;
    }

    if (auto it = std::dynamic_pointer_cast<sequence_iterator_value<false>>(source))
    {
        return ++*it, it;
    }

    if (auto it = std::dynamic_pointer_cast<generator_value>(source))
    {
        return it->next();
    }

    throw std::exception("Unknown next source");
}

auto len_iterator()
{
    return overloaded
    {
        [](const std::shared_ptr<const sequence_value>& ptr)
        {
            return ::len(std::static_pointer_cast<const value>(ptr));
        },
        [](const std::shared_ptr<sequence_value>& ptr)
        {
            return ::len(std::static_pointer_cast<value>(ptr));
        }
    };
}

template <typename V, bool Const, typename T>
decltype(auto) smart_cast(T&& ptr)
{
    return std::dynamic_pointer_cast<std::conditional_t<Const, std::add_const_t<V>, std::remove_const_t<V>>>(std::forward<T>(ptr));
}

template <typename T>
std::optional<std::size_t> len(T&& source)
{
    using source_t = decltype(source);
    static_assert(!std::is_pointer_v<source_t>);
    static_assert(is_smart_pointer_v<std::remove_cvref_t<source_t>>);
    using source_inner_t = type_decay_ptr_t<std::remove_cvref_t<source_t>>;
    constexpr auto source_inner_const = std::is_const_v<source_inner_t>;

#define CAST(to_type) (smart_cast<to_type, source_inner_const>(std::forward<T>(source)))  // NOLINT(cppcoreguidelines-macro-usage)

    if (const auto it = CAST(sequence_iterator_value<true>))
    {
        return std::visit(
            len_iterator(),
            it->sequence
        );
    }

    if (const auto it = CAST(sequence_iterator_value<false>))
    {
        return std::visit(
            len_iterator(),
            it->sequence
        );
    }

    if (const auto it = CAST(generator_value))
    {
        std::shared_ptr<sequence_value> seq;
        if constexpr (source_inner_const)
        {
            seq = value::create<sequence_value>(std::const_pointer_cast<generator_value>(it));
        }
        else
        {
            seq = value::create<sequence_value>(it);
        }

        return len(seq);
    }

    if (const auto it = CAST(sequence_value))
    {
        it->drain_generator();
        return it->values.size();
    }

#undef CAST

    return std::nullopt;
}

void bind(const std::shared_ptr<value> expression, const std::shared_ptr<value> source)
{
    auto expression_variable = std::dynamic_pointer_cast<variable>(expression);
    auto expression_seq = std::dynamic_pointer_cast<sequence_value>(expression);
    auto source_variable = std::dynamic_pointer_cast<variable>(source);
    auto source_seq = std::dynamic_pointer_cast<sequence_value>(source);

    if (expression_variable && source_variable)
    {
        expression_variable->value = source_variable->value;
        return;
    }

    if (expression_seq && source_seq)
    {
        const auto expression_len = len(expression);
        const auto source_len = len(source);
        if (expression_len == source_len)
        {
            for (std::size_t i = 0; i < expression_len; ++i)
            {
                ::bind(expression_seq->values[i], source_seq->values[i]);
            }
            return;
        }
    }

    if (expression_variable)
    {
        expression_variable->value = source;
        return;
    }

    throw std::exception("Cannot bind");
}

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

variable_late_binding::variable_late_binding(std::shared_ptr<variable> bound_variable)
    : name(bound_variable->name),
      bound_variable(std::move(bound_variable))
{
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
    return body->evaluate();
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

sequence_value::iterator sequence_value::begin()
{
    return iterator(shared_from_this());
}

sequence_value::iterator sequence_value::end()
{
    return iterator(shared_from_this(), sequence_iterator_end);
}

sequence_value::const_iterator sequence_value::begin() const
{
    return cbegin();
}

sequence_value::const_iterator sequence_value::end() const
{
    return cend();
}

sequence_value::const_iterator sequence_value::cbegin() const
{
    return const_iterator(shared_from_this());
}

sequence_value::const_iterator sequence_value::cend() const
{
    return const_iterator(shared_from_this(), sequence_iterator_end);
}

sequence_value::size_type sequence_value::size() const
{
    return std::distance(cbegin(), cend());
}

std::shared_ptr<value> sequence_value::evaluate()
{
    decltype(values) result;

    for (auto&& item : *this)
    {
        if (item)
        {
            result.push_back(item->evaluate());
        }
        else
        {
            throw std::exception("NullReference");
        }
    }

    return create<sequence_value>(result);
}

void sequence_value::drain_generator() const
{
    if (!generator)
        return;

    while (auto var = generator->next())
    {
        values.push_back(var);
    }
}

format_context& sequence_value::format(format_context& output) const
{
    this->drain_generator();

    auto first = true;
    for (auto&& value : *this)
    {
        if (!first)
            output << ' ';
        else
            first = false;

        output << value;
    }

    return output;
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

std::shared_ptr<value> parse_into_ast(const std::shared_ptr<ast_tree_context>& ast_context, const std::shared_ptr<peg::Ast>& ast)
{
    auto&& node_name = ast->name;

    if (node_name == "expression" && ast->nodes.size() == 1)
    {
        return parse_into_ast(ast_context, ast->nodes.front());
    }

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

        variable->value = parse_into_ast(context, nodes[1]);
        return parse_into_ast(context, nodes[2]);
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
        return parse_into_ast(ast_context, nodes[0]);
    }

    if (node_name == "if")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 4, assert_module{});
        DEBUG_ASSERT(nodes[0]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[2]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[3]->original_name == "expression", assert_module{});

        auto&& left = parse_into_ast(ast_context, nodes[0]);
        auto&& right = parse_into_ast(ast_context, nodes[1]);
        auto&& suite_true = parse_into_ast(ast_context, nodes[2]);
        auto&& suite_false = parse_into_ast(ast_context, nodes[3]);

        return std::static_pointer_cast<value>(value::create<condition_value>(left, condition_operator_kind::greater, right, suite_true, suite_false));
    }

    if (node_name == "bind")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});

        auto&& inner_type = nodes[0]->name;

        if (inner_type == "ident")
        {
            return ast_context->name_lookup->define(nodes[0]->token);
        }

        DEBUG_ASSERT(inner_type == "tuple", assert_module{});

        return parse_into_ast(ast_context, nodes[0]);
    }

    if (node_name == "tuple")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(!nodes.empty(), assert_module{});

        std::deque<sequence_value::value_type> values;

        for (auto&& ast_argument : nodes)
        {
            values.push_back(parse_into_ast(ast_context, ast_argument));
        }

        return value::create<sequence_value>(values);
    }

    if (node_name == "function")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "bind", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});

        const auto context = context_base::create<ast_tree_context>(ast_context);

        std::shared_ptr<value> args = parse_into_ast(context, nodes[0]);

        if (!static_cast<bool>(std::dynamic_pointer_cast<sequence_value>(args)))
        {
            const std::deque<sequence_value::value_type> values{args};
            args = value::create<sequence_value>(values);
        }

        auto&& body = parse_into_ast(context, nodes[1]);

        return value::create<managed_function_value>(args, context->name_lookup, body);
    }

    if (node_name == "call")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});

        auto&& what = parse_into_ast(ast_context, nodes[0]);
        auto&& with = parse_into_ast(ast_context, nodes[1]);

        const std::deque<sequence_value::value_type> values{with};
        auto&& args = value::create<sequence_value>(values);

        return value::create<call_value>(what, args);
    }

    if (node_name == "free_call")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(!nodes.empty(), assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "ident", assert_module{});

        auto&& identifier = nodes[0]->token;

        auto var = (*ast_context->name_lookup)[identifier]->bind(ast_context->name_lookup).bound_variable;
        if (var)
        {
            std::deque<sequence_value::value_type> values;

            // When will <range> finally arrive?
            auto first = true;
            for (auto&& ast_argument : nodes)
            {
                if (first)
                {
                    // So uncivilized.
                    first = false;
                    continue;
                }

                values.push_back(parse_into_ast(ast_context, ast_argument));
            }

            auto&& args = value::create<sequence_value>(values);

            return value::create<call_value>(var, args);
        }
    }

    DEBUG_UNREACHABLE(assert_module{});

    return nullptr;
}

std::shared_ptr<value> evaluate(const std::shared_ptr<value>& ast)
{
    return ast->evaluate();
}

std::string str(std::shared_ptr<value> const& expression, const bool force_explicit_call = false)
{
    if (!expression)
        return "nullptr";

    std::ostringstream buffer;

    format_context context(buffer);
    context.set_force_explicit_call(force_explicit_call);
    expression->format(context);

    return buffer.str();
}

std::shared_ptr<value> peg_parser(const std::string& source)
{
    peg::parser parser;

    DEBUG_ASSERT(parser.load_grammar(grammar), assert_module{});

    parser.enable_ast();

    std::shared_ptr<peg::Ast> ast;
    if (!parser.parse(source.c_str(), ast))
        return nullptr;

    const auto context = context_base::create<ast_tree_context>();
    context->name_lookup->define("add")->value = value::create<add_function_value>(context->name_lookup);
    return parse_into_ast(context, ast);
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

    try
    {
        const auto ast = peg_parser(source);
        const auto result = evaluate(ast);
        fout << str(result) << std::endl;
    }
    catch (...)
    {
        fout << "ERROR" << std::endl;
    }

    return 0;
}

std::shared_ptr<value> call_value::evaluate()
{
    const auto what_binding = std::dynamic_pointer_cast<variable_late_binding>(this->what);
    const auto what_var = what_binding ? what_binding->bound_variable : std::dynamic_pointer_cast<variable>(this->what);
    const auto what = what_var ? what_var->value : this->what;

    const auto function = std::dynamic_pointer_cast<function_value>(what);
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

bool condition_value::evaluate_condition() const
{
    auto&& left = this->left->evaluate();
    auto&& right = this->right->evaluate();
    const auto result = compare(left.get(), right.get());
    switch (this->kind)
    {
        case condition_operator_kind::equal:
            return std::is_eq(result);
        case condition_operator_kind::not_equal:
            return std::is_neq(result);
        case condition_operator_kind::less:
            return std::is_lt(result);
        case condition_operator_kind::less_equal:
            return std::is_lteq(result);
        case condition_operator_kind::greater:
            return std::is_gt(result);
        case condition_operator_kind::greater_equal:
            return std::is_gteq(result);
        default:
            throw std::exception("Unknown comparison kind");
    }
}
#pragma warning( push )
#pragma warning( disable : 26444 )
void test_suite()
{
    using namespace boost::ut;
    using namespace std::literals;

    "examples"_test = [] {
        "1"_test = [] {
            const auto ast = peg_parser(
                R"(

(let K = (val 10) in
                                              (add
                                                  (val 5)
                                                  (var K)))
)");

            const auto eval = evaluate(ast);

            expect(eq(str(eval), "(val 15)"s));
        };
        "2"_test = [] {
            const auto ast = peg_parser(
                R"(

(let A = (val 20) in
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
)

                )"
            );

            const auto eval = evaluate(ast);

            expect(eq(str(eval), "(val 31)"s));
        };
        "3"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function arg (add (var arg) (val 1))) in
    (let V = (val -1) in
        (call (var F) (var V))
    )
)

                )"
            );

            const auto eval = evaluate(ast);

            expect(eq(str(eval), "(val 0)"s));
        };
        "4"_test = [] {
            const auto ast = peg_parser(
                R"(

(add (var A) (var B))

                )"
            );

            expect(throws<std::exception>([&ast] { evaluate(ast); }));
        };
    };

    "extensions"_test = [] {
        "1"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function arg (add (var arg) (val 1))) in
    (var F))

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);
            const auto evaluated_fun = std::dynamic_pointer_cast<function_value>(evaluated);
            expect(static_cast<bool>(evaluated_fun) == true_b);

            expect(eq(str(evaluated, false), "(function arg (add (var arg) (val 1)))"s));
            expect(eq(str(evaluated, true), "(function arg (add (var arg) (val 1)))"s));
        };
        "2"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function arg (add (var arg) (val 1) (var arg))) in
    (F (val 15))
)

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "(val 31)"s));
            expect(eq(str(evaluated, true), "(val 31)"s));
        };
        "3"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function [x,y] (add (var x) (val 1) (var y))) in
    (F (val 15) (val 30))
)

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "(val 46)"s));
            expect(eq(str(evaluated, true), "(val 46)"s));
        };
    };
}
#pragma warning( pop )
