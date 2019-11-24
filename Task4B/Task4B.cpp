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
#include "igor.hpp"

// cppppack: embed point

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

expression   <- _ (val / var / if / let / function / call / free_call / list / generator) (ternary)? _
val          <- '(' 'val' integer ')'
var          <- '(' 'var' ident ')'
if           <- '(' 'if' expression expression 'then' expression 'else' expression ')'
let          <- '(' 'let' ident '=' expression 'in' expression ')'
function     <- '(' 'function' bind expression ')'
call         <- '(' 'call' expression expression ')'
free_call    <- '(' ident expression* ')'
list         <- '[' expression (',' expression)* ']'
generator    <- 'for' bind 'in' expression ':' expression
ternary      <- 'if' condition (ternary_else)?
ternary_else <- 'else' expression

condition  <- expression compare_op expression

compare_op <- _ < '==' / '!=' / '<=' / '<' / '>=' / '>' > _
bind       <- (tuple / ident)
ident      <- _ < (([a-zA-Z] [a-zA-Z0-9]*) / '+' / '-' / '*' / '/' / '%') > _
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
struct name_lookup_context;

struct value;
struct integer_value;
struct function_value;
struct generator_value;
struct sequence_value;
struct sequence_bound_view;

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

template <typename T>
std::optional<std::size_t> len(const std::shared_ptr<name_lookup_context>& context, T&& source);

void bind(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<value> expression, std::shared_ptr<value> source);

std::shared_ptr<value> parse_into_ast(const std::shared_ptr<peg::Ast>& ast);
std::shared_ptr<value> parse_bind(const std::shared_ptr<name_lookup_context>& name_lookup_context, const std::shared_ptr<peg::Ast>& ast);


struct parameter_pack_tag
{
};

inline constexpr auto parameter_pack = igor::named_argument<parameter_pack_tag>{};

struct sequence_as_args_tag
{
};

inline constexpr auto sequence_as_args = igor::named_argument<sequence_as_args_tag>{};

struct define_tag
{
};

inline constexpr auto define = igor::named_argument<define_tag>{};

template <typename T, typename ... Args>
format_context& format_scope(format_context& output, T&& value, Args&& ... args);

struct value
{
    virtual ~value() = default;

    virtual std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) = 0;
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
    bool force_explicit_call_ = false;

public:
    [[nodiscard]] bool force_explicit_call() const
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
                DEBUG_ASSERT(static_cast<bool>(obj), assert_module{});
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

    [[nodiscard]] bool sequence_as_args() const
    {
        const auto scope = current_scope();
        return scope ? scope->sequence_as_args() : false;
    }

    [[nodiscard]] std::shared_ptr<name_lookup_context> name_lookup_context() const
    {
        const auto scope = current_scope();
        return scope ? scope->name_lookup_context() : nullptr;
    }

    class scope final
    {
        format_context* const context_;
        scope* const parent_;
        std::optional<std::int16_t> precedence_;
        bool wrap_in_brackets_ = false;
        std::shared_ptr<::name_lookup_context> name_lookup_context_;
        std::optional<bool> parameter_pack_;
        std::optional<bool> sequence_as_args_;

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

        [[nodiscard]] bool sequence_as_args() const
        {
            if (sequence_as_args_.has_value())
            {
                return sequence_as_args_.value();
            }

            auto* parent = parent_;
            while (parent)
            {
                if (parent->sequence_as_args_.has_value())
                {
                    return parent->sequence_as_args_.value();
                }

                parent = parent->parent_;
            }

            return false;
        }

        void set_sequence_as_args(const bool sequence_as_args)
        {
            sequence_as_args_ = sequence_as_args;
        }

        [[nodiscard]] std::shared_ptr<::name_lookup_context> name_lookup_context() const
        {
            if (name_lookup_context_)
            {
                return name_lookup_context_;
            }

            auto* parent = parent_;
            while (parent)
            {
                if (parent->name_lookup_context_)
                {
                    return parent->name_lookup_context_;
                }

                parent = parent->parent_;
            }

            return nullptr;
        }

        void set_name_lookup_context(const std::shared_ptr<::name_lookup_context>& name_lookup_context)
        {
            name_lookup_context_ = name_lookup_context;
        }

        scope(const scope& other) = delete;
        scope(scope&& other) noexcept = delete;
        scope& operator=(const scope& other) = delete;
        scope& operator=(scope&& other) noexcept = delete;
    };
};

template <typename T>
void format_scope_define_impl(const format_context::scope& scope, T&& name);

struct variable_late_binding final : value
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

    friend std::size_t hash_value(const variable_late_binding& obj) noexcept
    {
        std::size_t seed = 0x32ECF885;
        seed ^= (seed << 6) + (seed >> 2) + 0x161D3D4A + std::hash<decltype(name)>()(obj.name);
        return seed;
    }

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override;

    format_context& format(format_context& output) const override
    {
        if (output.parameter_pack())
            return output << name;

        return output << "(var " << name << ")";
    }

private:
    mutable std::pair<std::shared_ptr<name_lookup_context>, std::shared_ptr<variable>> bound_variable_;

    explicit variable_late_binding(std::string name)
        : name(std::move(name))
    {
    }

    void bind(const std::shared_ptr<name_lookup_context>& context) const;

    friend struct value;
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

struct let_value final : value
{
    std::string name;
    std::shared_ptr<value> new_value;
    std::shared_ptr<value> body;

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override;

    format_context& format(format_context& output) const override
    {
        return output << "(let " << name << " = " << new_value << " in " << format_scope(output, body, define = name) << ')';
    }

private:
    let_value(std::string name, std::shared_ptr<value> new_value, std::shared_ptr<value> value)
        : name(std::move(name)),
          new_value(std::move(new_value)),
          body(std::move(value))
    {
    }

    friend struct value;
};

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

    std::shared_ptr<::value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
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
    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
    {
        return shared_from_this();
    }

    std::shared_ptr<value> execute(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments)
    {
        return execute_impl(context, std::move(arguments));
    }

protected:
    virtual std::shared_ptr<value> execute_impl(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments) = 0;
};

struct native_function_value : function_value
{
    format_context& format(format_context& output) const override
    {
        throw std::exception("Shouldn't be formatted");
    }

protected:
    explicit native_function_value() = default;
};

struct accumulator_function_value : native_function_value
{
    std::shared_ptr<value> execute_impl(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments) override;

    virtual void accumulate(int64_t& accumulator, int64_t value) const = 0;

protected:
    explicit accumulator_function_value() = default;
};

struct add_function_value final : accumulator_function_value
{
    void accumulate(int64_t& accumulator, const int64_t value) const override
    {
        accumulator += value;
    }

private:
    explicit add_function_value() = default;

    friend struct value;
};

struct sub_function_value final : accumulator_function_value
{
    void accumulate(int64_t& accumulator, const int64_t value) const override
    {
        accumulator -= value;
    }

private:
    explicit sub_function_value() = default;

    friend struct value;
};

struct mul_function_value final : accumulator_function_value
{
    void accumulate(int64_t& accumulator, const int64_t value) const override
    {
        accumulator *= value;
    }

private:
    explicit mul_function_value() = default;

    friend struct value;
};

struct div_function_value final : accumulator_function_value
{
    void accumulate(int64_t& accumulator, const int64_t value) const override
    {
        accumulator /= value;
    }

private:
    explicit div_function_value() = default;

    friend struct value;
};

struct rem_function_value final : accumulator_function_value
{
    void accumulate(int64_t& accumulator, const int64_t value) const override
    {
        accumulator %= value;
    }

private:
    explicit rem_function_value() = default;

    friend struct value;
};

struct managed_function_value final : function_value
{
    std::shared_ptr<peg::Ast> argument; // delayed variable creation
    std::shared_ptr<value> body;

    std::shared_ptr<value> execute_impl(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments) override;

    format_context& format(format_context& output) const override;

    std::shared_ptr<value> materialize_arguments(const std::shared_ptr<name_lookup_context>& context) const;

private:
    managed_function_value(std::shared_ptr<peg::Ast> argument, std::shared_ptr<value> value)
        : argument(std::move(argument)),
          body(std::move(value))
    {
    }

    friend struct value;
};

struct generator_value : value, std::enable_shared_from_this<generator_value>
{
    virtual std::shared_ptr<value> next(const std::shared_ptr<name_lookup_context>& context) = 0;
};

struct native_generator_value : generator_value
{
    format_context& format(format_context& output) const override
    {
        throw std::exception("Shouldn't be formatted");
    }

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
    {
        return shared_from_this();
    }
};

struct sequence_value final : value, std::enable_shared_from_this<sequence_value>
{
    using value_type = std::shared_ptr<value>;

    mutable std::deque<value_type> values;
    std::shared_ptr<generator_value> generator;

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override;

    void drain_generator(const std::shared_ptr<name_lookup_context>& context) const;

    format_context& format(format_context& output) const override;

    std::shared_ptr<sequence_bound_view> view(const std::shared_ptr<name_lookup_context>& context) const;

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

struct sequence_bound_view final : std::enable_shared_from_this<sequence_bound_view>
{
    using value_type = std::shared_ptr<value>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using const_iterator = sequence_iterator_value<true>;
    using iterator = sequence_iterator_value<false>;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    std::shared_ptr<name_lookup_context> context;
    std::variant<std::shared_ptr<const sequence_value>, std::shared_ptr<sequence_value>> sequence;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    const_iterator cbegin() const;
    const_iterator cend() const;

    size_type size() const;

    void generate_next() const
    {
        ::std::visit(
            [this](auto&& ptr)
            {
                auto&& generator = ptr->generator;
                if (generator)
                {
                    const auto next = generator->next(context);
                    if (next)
                    {
                        ptr->values.push_back(next);
                    }
                }
            },
            sequence
        );
    }

    template <typename... Args>
    static decltype(auto) create(Args&&... params)
    {
        // ReSharper disable once CppSmartPointerVsMakeFunction
        return std::shared_ptr<sequence_bound_view>(::new sequence_bound_view(std::forward<Args>(params)...));
    }

private:
    sequence_bound_view(std::shared_ptr<name_lookup_context> name_lookup_context,
                        std::shared_ptr<const sequence_value> sequence)
        : context(std::move(name_lookup_context)),
          sequence(std::move(sequence))
    {
    }

    sequence_bound_view(std::shared_ptr<name_lookup_context> name_lookup_context,
                        std::shared_ptr<sequence_value> sequence)
        : context(std::move(name_lookup_context)),
          sequence(std::move(sequence))
    {
    }
};

template <bool Const>
struct sequence_iterator_value final
{
    using value_type_const = std::conditional_t<Const, const value, value>;
    using value_type = std::shared_ptr<value_type_const>;
    using reference = value_type&;
    using pointer = value_type*;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    std::shared_ptr<const sequence_bound_view> sequence_view;
    std::size_t index;
    bool end;

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

    auto operator++(int)
    {
        auto ip = *this;
        ++*this;
        return ip;
    }

    auto operator--(int)
    {
        auto ip = *this;
        --*this;
        return ip;
    }

    ~sequence_iterator_value() = default;
    sequence_iterator_value(const sequence_iterator_value& other) = default;
    sequence_iterator_value(sequence_iterator_value&& other) noexcept = default;
    sequence_iterator_value& operator=(const sequence_iterator_value& other) = default;
    sequence_iterator_value& operator=(sequence_iterator_value&& other) noexcept = default;

private:
    explicit sequence_iterator_value(std::shared_ptr<sequence_bound_view> sequence_value)
        : sequence_view(std::move(sequence_value)), index(0u), end(false)
    {
        update_end_state();
    }

    explicit sequence_iterator_value(std::shared_ptr<const sequence_bound_view> sequence_value)
        : sequence_view(std::move(sequence_value)), index(0u), end(false)
    {
        update_end_state();
    }

    explicit sequence_iterator_value(std::shared_ptr<sequence_bound_view> sequence_value, sequence_iterator_end_t)
        : sequence_view(std::move(sequence_value)), index(std::numeric_limits<decltype(index)>::max()), end(true)
    {
        update_end_state();
    }

    explicit sequence_iterator_value(std::shared_ptr<const sequence_bound_view> sequence_value, sequence_iterator_end_t)
        : sequence_view(std::move(sequence_value)), index(std::numeric_limits<decltype(index)>::max()), end(true)
    {
        update_end_state();
    }

    void update_end_state()
    {
        if (!end && index >= std::visit(
            [](auto&& ptr)
            {
                return ptr->values;
            },
            sequence_view->sequence
        ).size())
        {
            sequence_view->generate_next();
        }

        end = index >= std::visit(
            [](auto&& ptr)
            {
                return ptr->values;
            },
            sequence_view->sequence
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
                sequence_view->sequence
            )
            [index];
    }

    friend struct value;
    friend struct sequence_bound_view;
};

struct managed_generator_value final : generator_value
{
private:
    struct sequence_cache
    {
        std::shared_ptr<name_lookup_context> context;
        std::shared_ptr<value> source;
        std::shared_ptr<sequence_bound_view> bound_view;
        std::optional<sequence_iterator_value<false>> iterator;
    };

public:
    // for x in args: f(x)                                  => x
    // for x in args: f(x) if predicate(x)                  => x
    // for [i, x] in enumerate(args): f(x) if predicate(x)  => [i, x]
    std::shared_ptr<peg::Ast> expression;
    // for x in args: f(x)                                  => args
    // for x in args: f(x) if predicate(x)                  => args
    // for [i, x] in enumerate(args): f(x) if predicate(x)  => enumerate(args)
    std::shared_ptr<value> source;
    // for x in args: f(x)                                  => f(x)
    // for x in args: f(x) if predicate(x)                  => f(x) if predicate(x)
    // for [i, x] in enumerate(args): f(x) if predicate(x)  => f(x) if predicate(x)
    std::shared_ptr<value> transform;

    std::shared_ptr<value> next(const std::shared_ptr<name_lookup_context>& context);

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
    {
        evaluate_source(context);
        return shared_from_this();
    }

    format_context& format(format_context& output) const override;

private:
    sequence_cache source_evaluated_;

    void evaluate_source(const std::shared_ptr<name_lookup_context>& context)
    {
        source_evaluated_.context = context;
        source_evaluated_.source = source->evaluate(context);
        source_evaluated_.iterator = std::nullopt;
        source_evaluated_.bound_view = nullptr;
    }

    managed_generator_value(std::shared_ptr<peg::Ast> expression, std::shared_ptr<value> source,
        std::shared_ptr<value> transform)
        : expression(std::move(expression)),
        source(std::move(source)),
        transform(std::move(transform))
    {
    }

    friend struct value;
};

template <bool Const1, bool Const2>
bool operator==(const sequence_iterator_value<Const1>& lhs, const sequence_iterator_value<Const2>& rhs)
{
    return lhs.sequence_view == rhs.sequence_view
        && (lhs.index == rhs.index || (lhs.end && rhs.end));
}

template <bool Const1, bool Const2>
bool operator!=(const sequence_iterator_value<Const1>& lhs, const sequence_iterator_value<Const2>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, typename ... Args>
format_context& format_scope(format_context& output, T&& value, Args&& ... args)
{
    igor::parser p{args...};
    format_context::scope scope(output);

    if constexpr (p.has(parameter_pack))
    {
        DEBUG_ASSERT(!output.parameter_pack(), assert_module{});
        scope.set_parameter_pack(true);
    }

    if constexpr (p.has(sequence_as_args))
    {
        DEBUG_ASSERT(!output.sequence_as_args(), assert_module{});
        scope.set_sequence_as_args(true);
    }

    if constexpr (p.has(define))
    {
        auto name = p(define);
        format_scope_define_impl(scope, name);
    }

    return output << std::forward<decltype(value)>(value);
}

struct call_value final : value
{
    std::shared_ptr<value> what;
    std::shared_ptr<sequence_value> arguments;

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override;

    format_context& format(format_context& output) const override
    {
        output << '(';

        if (output.force_explicit_call() && len(output.name_lookup_context(), arguments) == 1)
            output << "call ";

        return output << format_scope(output, what, parameter_pack = true) << ' ' << format_scope(output, arguments, sequence_as_args = true) << ')';
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

    [[nodiscard]] bool evaluate_condition(const std::shared_ptr<name_lookup_context>& context) const;

    std::shared_ptr<value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
    {
        return evaluate_condition(context) ? suite_true->evaluate(context) : suite_false->evaluate(context);
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

    std::shared_ptr<::value> evaluate(const std::shared_ptr<name_lookup_context>& context) override
    {
        return value->evaluate(context);
    }

    format_context& format(format_context& output) const override
    {
        // Variables are stored only as part of binding expressions (function arguments/generator items)
        DEBUG_ASSERT(output.parameter_pack(), assert_module{});
        return output << name;
    }

private:
    explicit variable(std::string name)
        : name(std::move(name))
    {
    }

    friend struct value;
};

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

    std::shared_ptr<variable> define(const std::string& key)
    {
        return defined.emplace(key, value::create<variable>(key)).first->second;
    }

private:
    explicit name_lookup_context(std::shared_ptr<name_lookup_context> parent)
        : parent(std::move(parent))
    {
    }

    explicit name_lookup_context() = default;

    std::shared_ptr<variable> find_definition(const std::string& key)
    {
        const auto local_definition = defined.find(key);
        if (local_definition != defined.end())
        {
            return local_definition->second;
        }

        return parent ? parent->find_definition(key) : nullptr;
    }

    friend struct context_base;
    friend struct variable_late_binding;
};

template <typename T>
void format_scope_define_impl(const format_context::scope& scope, T&& name)
{
    auto context = scope.name_lookup_context();
    context->define(name);
}

void test_suite();

auto len_iterator(const std::shared_ptr<name_lookup_context>& context)
{
    return overloaded
    {
        [&context](const std::shared_ptr<const sequence_value>& ptr)
        {
            return ::len(context, std::static_pointer_cast<const value>(ptr));
        },
        [&context](const std::shared_ptr<sequence_value>& ptr)
        {
            return ::len(context, std::static_pointer_cast<value>(ptr));
        }
    };
}

template <typename V, bool Const, typename T>
decltype(auto) smart_cast(T&& ptr)
{
    return std::dynamic_pointer_cast<std::conditional_t<Const, std::add_const_t<V>, std::remove_const_t<V>>>(std::forward<T>(ptr));
}

template <typename T>
std::optional<std::size_t> len(const std::shared_ptr<name_lookup_context>& context, T&& source)
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
            len_iterator(context),
            it->sequence_view->sequence
        );
    }

    if (const auto it = CAST(sequence_iterator_value<false>))
    {
        return std::visit(
            len_iterator(context),
            it->sequence_view->sequence
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

        return len(context, seq);
    }

    if (const auto it = CAST(sequence_value))
    {
        it->drain_generator(context);
        return it->values.size();
    }

#undef CAST

    return std::nullopt;
}

void bind(const std::shared_ptr<name_lookup_context>& context, const std::shared_ptr<value> expression, const std::shared_ptr<value> source)
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
        const auto expression_len = len(context, expression);
        const auto source_len = len(context, source);
        if (expression_len == source_len)
        {
            for (std::size_t i = 0; i < expression_len; ++i)
            {
                ::bind(context, expression_seq->values[i], source_seq->values[i]);
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

void variable_late_binding::bind(const std::shared_ptr<name_lookup_context>& context) const
{
    if (bound_variable_.second && bound_variable_.first == context || !context)
        return;

    const auto variable = context->find_definition(name);
    if (bound_variable_.second != variable)
        bound_variable_ = std::make_pair(context, variable);
}

std::shared_ptr<value> let_value::evaluate(const std::shared_ptr<name_lookup_context>& context)
{
    context->define(name)->value = new_value;
    return body->evaluate(context);
}

std::shared_ptr<value> variable_late_binding::evaluate(const std::shared_ptr<name_lookup_context>& context)
{
    bind(context);
    return bound_variable_.second ? bound_variable_.second->evaluate(context) : nullptr;
}

std::shared_ptr<value> accumulator_function_value::execute_impl(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments)
{
    int64_t result = 0;

    const auto arguments_view = arguments->view(context);

    for (auto&& value : *arguments_view)
    {
        if (const auto integral = std::dynamic_pointer_cast<integer_value>(value))
        {
            accumulate(result, integral->value);
        }
        else
        {
            throw std::exception("add: expected integer arguments");
        }
    }

    return create<integer_value>(result);
}

std::shared_ptr<value> managed_generator_value::next(const std::shared_ptr<name_lookup_context>& context)
{
    DEBUG_ASSERT(source, assert_module{});

    {
        std::shared_ptr<value> source;

        do
        {
            if (source_evaluated_.context == context)
            {
                if (source_evaluated_.source)
                {
                    source = source_evaluated_.source;
                }
            }

            if (!source)
            {
                this->evaluate_source(context);
            }
        }
        while (!source);

        if (source_evaluated_.iterator)
        {
            const auto same_source = ::std::visit(
                [&source](auto&& ptr)
                {
                    return ptr == source;
                },
                source_evaluated_.iterator->sequence_view->sequence
            );

            if (!same_source)
            {
                source_evaluated_.iterator = std::nullopt;
            }
        }

        if (!source_evaluated_.iterator)
        {
            if (const auto seq = std::static_pointer_cast<sequence_value>(source))
            {
                source_evaluated_.bound_view = seq->view(context);
                source_evaluated_.iterator = source_evaluated_.bound_view->begin();
            }
        }
    }

    DEBUG_ASSERT(source_evaluated_.iterator, assert_module{});

    while (!source_evaluated_.iterator->end)
    {
        auto&& it = *source_evaluated_.iterator;
        auto item = *it;
        ++it;

        if (!item)
        {
            source_evaluated_.iterator = source_evaluated_.bound_view->end();
            return nullptr;
        }

        const std::deque<sequence_value::value_type> values{ item };
        item = create<sequence_value>(values);

        const auto local_context = context_base::create<name_lookup_context>(context);
        const auto argument = parse_bind(local_context, expression);
        ::bind(local_context, argument, item);

        auto transformed = transform->evaluate(local_context);

        if (!transformed)
        {
            continue;
        }

        return transformed;
    }

    return nullptr;
}

format_context& managed_generator_value::format(format_context& output) const
{
    const auto local_context = context_base::create<name_lookup_context>(output.name_lookup_context());

    format_context::scope scope(output);
    scope.set_name_lookup_context(local_context);

    const auto argument = parse_bind(local_context, expression);
    return output << "for " << format_scope(output, argument, parameter_pack = true, sequence_as_args = true) << " in " << source << ": " << transform;
}

static_assert(std::is_same_v<std::iterator_traits<sequence_bound_view::const_iterator>::difference_type, std::ptrdiff_t>);
static_assert(std::is_same_v<std::iterator_traits<sequence_bound_view::const_iterator>::iterator_category, std::bidirectional_iterator_tag>);

sequence_bound_view::iterator sequence_bound_view::begin()
{
    return iterator(shared_from_this());
}

sequence_bound_view::iterator sequence_bound_view::end()
{
    return iterator(shared_from_this(), sequence_iterator_end);
}

sequence_bound_view::const_iterator sequence_bound_view::begin() const
{
    return cbegin();
}

sequence_bound_view::const_iterator sequence_bound_view::end() const
{
    return cend();
}

sequence_bound_view::const_iterator sequence_bound_view::cbegin() const
{
    return const_iterator(shared_from_this());
}

sequence_bound_view::const_iterator sequence_bound_view::cend() const
{
    return const_iterator(shared_from_this(), sequence_iterator_end);
}

sequence_bound_view::size_type sequence_bound_view::size() const
{
    return std::distance(cbegin(), cend());
}

std::shared_ptr<value> sequence_value::evaluate(const std::shared_ptr<name_lookup_context>& context)
{
    decltype(values) result;

    const auto arguments_view = this->view(context);

    for (auto&& item : *arguments_view)
    {
        if (item)
        {
            result.push_back(item->evaluate(context));
        }
        else
        {
            throw std::exception("NullReference");
        }
    }

    return create<sequence_value>(result);
}

void sequence_value::drain_generator(const std::shared_ptr<name_lookup_context>& context) const
{
    if (!generator)
        return;

    while (auto var = generator->next(context))
    {
        values.push_back(var);
    }
}

format_context& sequence_value::format(format_context& output) const
{
    auto context = output.name_lookup_context();

    this->drain_generator(context);

    const auto sequence_as_args = output.sequence_as_args();

    if (!sequence_as_args)
        output << '[';

    const auto arguments_view = this->view(context);

    auto first = true;
    for (auto&& value : *arguments_view)
    {
        if (!first)
            output << (sequence_as_args ? " " : ", ");
        else
            first = false;

        output << value;
    }

    if (!sequence_as_args)
        output << ']';

    return output;
}

std::shared_ptr<sequence_bound_view> sequence_value::view(const std::shared_ptr<name_lookup_context>& context) const
{
    return sequence_bound_view::create(context, shared_from_this());
}

std::shared_ptr<value> managed_function_value::execute_impl(const std::shared_ptr<name_lookup_context>& context, std::shared_ptr<sequence_value> arguments)
{
    const auto local_context = context_base::create<name_lookup_context>(context);
    const auto argument = materialize_arguments(local_context);
    ::bind(local_context, argument, arguments);
    return body->evaluate(local_context);
}

format_context& managed_function_value::format(format_context& output) const
{
    const auto local_context = context_base::create<name_lookup_context>(output.name_lookup_context());

    format_context::scope scope(output);
    scope.set_name_lookup_context(local_context);

    const auto argument = materialize_arguments(local_context);
    return output << "(function " << format_scope(output, argument, parameter_pack = true, sequence_as_args = true) << " " << body << ")";
}

std::shared_ptr<value> managed_function_value::materialize_arguments(const std::shared_ptr<name_lookup_context>& context) const
{
    return parse_bind(context, argument);
}

std::shared_ptr<value> parse_bind_impl(const std::shared_ptr<name_lookup_context>& name_lookup_context, const std::shared_ptr<peg::Ast>& ast)
{
    auto&& node_name = ast->name;

    if (node_name == "expression" && ast->nodes.size() == 1)
    {
        return parse_bind_impl(name_lookup_context, ast->nodes.front());
    }

    if (node_name == "bind")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});

        auto&& inner_type = nodes[0]->name;

        if (inner_type == "ident")
        {
            return name_lookup_context->define(nodes[0]->token);
        }

        DEBUG_ASSERT(inner_type == "tuple", assert_module{});

        return parse_bind_impl(name_lookup_context, nodes[0]);
    }

    if (node_name == "tuple")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(!nodes.empty(), assert_module{});

        std::deque<sequence_value::value_type> values;

        for (auto&& ast_argument : nodes)
        {
            values.push_back(parse_bind_impl(name_lookup_context, ast_argument));
        }

        return value::create<sequence_value>(values);
    }

    DEBUG_UNREACHABLE(assert_module{});

    return nullptr;
}

std::shared_ptr<value> parse_bind(const std::shared_ptr<name_lookup_context>& context, const std::shared_ptr<peg::Ast>& ast)
{
    auto args = parse_bind_impl(context, ast);

    if (!static_cast<bool>(std::dynamic_pointer_cast<sequence_value>(args)))
    {
        const std::deque<sequence_value::value_type> values{ args };
        args = value::create<sequence_value>(values);
    }

    return args;
}

std::shared_ptr<value> parse_into_ast(const std::shared_ptr<peg::Ast>& ast)
{
    auto&& node_name = ast->name;

    if (node_name == "expression" && ast->nodes.size() == 1)
    {
        return parse_into_ast(ast->nodes.front());
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
        auto&& value = parse_into_ast(nodes[1]);
        auto&& body = parse_into_ast(nodes[2]);

        return value::create<let_value>(identifier, value, body);
    }

    if (node_name == "var")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "ident", assert_module{});
        auto&& identifier = nodes[0]->token;

        return value::create<variable_late_binding>(identifier);
    }

    if (node_name == "val")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 1, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "integer", assert_module{});
        return parse_into_ast(nodes[0]);
    }

    if (node_name == "if")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 4, assert_module{});
        DEBUG_ASSERT(nodes[0]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[2]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[3]->original_name == "expression", assert_module{});

        auto&& left = parse_into_ast(nodes[0]);
        auto&& right = parse_into_ast(nodes[1]);
        auto&& suite_true = parse_into_ast(nodes[2]);
        auto&& suite_false = parse_into_ast(nodes[3]);

        return std::static_pointer_cast<value>(value::create<condition_value>(left, condition_operator_kind::greater, right, suite_true, suite_false));
    }

    if (node_name == "function")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "bind", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});

        auto&& body = parse_into_ast(nodes[1]);

        return value::create<managed_function_value>(nodes[0], body);
    }

    if (node_name == "call")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});

        auto&& what = parse_into_ast(nodes[0]);
        auto&& with = parse_into_ast(nodes[1]);

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

        auto var = value::create<variable_late_binding>(identifier);
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

                values.push_back(parse_into_ast(ast_argument));
            }

            auto&& args = value::create<sequence_value>(values);

            return value::create<call_value>(var, args);
        }
    }

    if (node_name == "list")
    {
        auto&& nodes = ast->nodes;
        if (nodes.size() == 1 && nodes[0]->name == "expression" && nodes[0]->nodes.size() == 1 && nodes[0]->nodes[0]->name == "generator")
        {
            auto&& generator = std::static_pointer_cast<managed_generator_value>(parse_into_ast(nodes[0]->nodes[0]));
            return value::create<sequence_value>(generator);
        }
        else
        {
            std::deque<sequence_value::value_type> values;

            for (auto&& ast_argument : nodes)
            {
                values.push_back(parse_into_ast(ast_argument));
            }

            return value::create<sequence_value>(values);
        }
    }

    if (node_name == "generator")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 3, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "bind", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});
        DEBUG_ASSERT(nodes[2]->original_name == "expression", assert_module{});

        auto&& source = parse_into_ast(nodes[1]);
        auto&& transform = parse_into_ast(nodes[2]);

        return value::create<managed_generator_value>(nodes[0], source, transform);
    }

    DEBUG_UNREACHABLE(assert_module{});

    return nullptr;
}

struct range_generator_value final : native_generator_value
{
    int64_t start;
    int64_t stop;
    int64_t step;
    int64_t i = 0;

    std::shared_ptr<value> next(const std::shared_ptr<name_lookup_context>& context) override
    {
        const auto result = start + step * i;
        if (i < 0)
        {
            return nullptr;
        }

        if (step > 0 && result >= stop)
        {
            return nullptr;
        }

        if (step < 0 && result <= stop)
        {
            return nullptr;
        }

        ++i;

        return create<integer_value>(result);
    }

private:
    range_generator_value(const int64_t start, const int64_t stop, const int64_t step = 1)
        : start(start),
          stop(stop),
          step(step)
    {
        if (step == 0)
        {
            throw std::domain_error("range step should be non-zero");
        }
    }

    friend struct value;
};

struct range_function_value final : native_function_value
{
protected:
    std::shared_ptr<value> execute_impl(const std::shared_ptr<name_lookup_context>& context,
                                        std::shared_ptr<sequence_value> arguments) override
    {
        const auto args_view = arguments->view(context);
        std::vector<std::shared_ptr<value>> args{args_view->begin(), args_view->end()};

        if (args.empty())
        {
            throw std::domain_error("range should specify at least stop point");
        }

        int64_t start = 0, stop, step = 1;
        switch (args.size())
        {
            case 1:
            {
                stop = evaluate_point(context, args[0]);
                break;
            }
            case 2:
            {
                start = evaluate_point(context, args[0]);
                stop = evaluate_point(context, args[1]);
                break;
            }
            case 3:
            {
                start = evaluate_point(context, args[0]);
                stop = evaluate_point(context, args[1]);
                step = evaluate_point(context, args[2]);
                break;
            }
            default:
            {
                throw std::domain_error("range expects no more than 3 arguments");
            }
        }

        const auto generator = create<range_generator_value>(start, stop, step);

        return create<sequence_value>(generator);
    }

    int64_t evaluate_point(const std::shared_ptr<name_lookup_context>& context,
                                        const std::shared_ptr<value>& argument) const
    {
        if (!argument)
        {
            throw std::exception("point is nullptr");
        }

        const auto evaluated = argument->evaluate(context);

        if (!evaluated)
        {
            throw std::exception("evaluated point is nullptr");
        }

        const auto integer = std::static_pointer_cast<integer_value>(evaluated);

        if (!integer)
        {
            throw std::exception("evaluated point is not an integer");
        }

        return integer->value;
    }

private:
    explicit range_function_value() = default;

    friend struct value;
};

std::shared_ptr<name_lookup_context> create_context()
{
    const auto context = context_base::create<name_lookup_context>();

    const auto add_function = value::create<add_function_value>();
    context->define("add")->value = add_function;
    context->define("+")->value = add_function;
    context->define("-")->value = value::create<sub_function_value>();
    context->define("*")->value = value::create<mul_function_value>();
    context->define("/")->value = value::create<div_function_value>();
    context->define("%")->value = value::create<rem_function_value>();
    context->define("range")->value = value::create<range_function_value>();

    return context;
}

std::shared_ptr<value> evaluate(const std::shared_ptr<value>& ast)
{
    const auto context = create_context();
    DEBUG_ASSERT(ast, assert_module{});
    return ast->evaluate(context);
}

std::string str(std::shared_ptr<value> const& expression, const bool force_explicit_call = false)
{
    if (!expression)
        return "nullptr";

    std::ostringstream buffer;

    const auto name_lookup_context = create_context();

    format_context context(buffer);
    context.set_force_explicit_call(force_explicit_call);

    format_context::scope scope(context);
    scope.set_name_lookup_context(name_lookup_context);

    context << expression;

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

    return parse_into_ast(ast);
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

std::shared_ptr<value> call_value::evaluate(const std::shared_ptr<name_lookup_context>& context)
{
    const auto what = this->what->evaluate(context);

    const auto function = std::dynamic_pointer_cast<function_value>(what);
    if (!function)
    {
        throw std::exception("Can only call functions");
    }

    const auto arguments = std::dynamic_pointer_cast<sequence_value>(this->arguments->evaluate(context));

    if (!arguments)
    {
        throw std::exception("Can only use sequences as arguments");
    }

    return function->execute(context, arguments);
}

bool condition_value::evaluate_condition(const std::shared_ptr<name_lookup_context>& context) const
{
    auto&& left = this->left->evaluate(context);
    auto&& right = this->right->evaluate(context);
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
        (var K)
    )
)

                )"
            );

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
        "4"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function [x,y] (+ (var x) (val 1) (var y))) in
    (F (val 15) (val 30))
)

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "(val 46)"s));
            expect(eq(str(evaluated, true), "(val 46)"s));
        };
        "5"_test = [] {
            const auto ast = peg_parser(
                R"(

[(val 1), (val 2), (val 3)]

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "[(val 1), (val 2), (val 3)]"s));
            expect(eq(str(evaluated, true), "[(val 1), (val 2), (val 3)]"s));
        };
        "6"_test = [] {
            const auto ast = peg_parser(
                R"(

[for x in (range (val 1) (val 5)): (var x)]

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "[(val 1), (val 2), (val 3), (val 4)]"s));
            expect(eq(str(evaluated, true), "[(val 1), (val 2), (val 3), (val 4)]"s));
        };
        skip | "7"_test = [] {
            const auto ast = peg_parser(
                R"(

[for x in (range (val 1) (val 10)): (var x) if (% (var x) (val 2)) == (val 0)]

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "[(val 2), (val 4), (val 6), (val 8)]"s));
            expect(eq(str(evaluated, true), "[(val 2), (val 4), (val 6), (val 8)]"s));
        };
        "8"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function C (if (var C) (val 10) then (var C) else (add (var C) (call (var F) (add (val 1) (var C)))))) in
    (call (var F) (val 0))
)

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "(val 66)"s));
            expect(eq(str(evaluated, true), "(val 66)"s));
        };
        "9"_test = [] {
            const auto ast = peg_parser(
                R"(

(let F = (function C (if (var C) (val 10) then (var C) else (add (call (var F) (add (val 1) (var C))) (var C)))) in
    (call (var F) (val 0))
)

                )"
            );

            const auto evaluated = evaluate(ast);
            expect(static_cast<bool>(evaluated) == true_b);

            expect(eq(str(evaluated, false), "(val 66)"s));
            expect(eq(str(evaluated, true), "(val 66)"s));
        };
    };
}
#pragma warning( pop )
