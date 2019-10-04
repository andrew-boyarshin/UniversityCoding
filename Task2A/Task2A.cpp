#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <system_error>
#include <type_traits>
#include <cassert>
#include <utility>
#include <stack>
#include <ostream>
#include <typeindex>
#include <cctype>
#include <vector>
#include <memory>
#include <limits>

#ifdef DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

// Only one per TEST_CASE is possible! (due to the use of static variables)
#define DOCTEST_INDEX_PARAMETERIZED_DATA(data, from, to)                                    \
    const std::size_t _doctest_subcase_count = (to) - (from) + 1;                           \
    static std::vector<std::string> _doctest_subcases = [&_doctest_subcase_count]() {       \
        std::vector<std::string> out;                                                       \
        while(out.size() != _doctest_subcase_count)                                         \
            out.push_back(std::string(#data " := ") + std::to_string((from) + out.size())); \
        return out;                                                                         \
    }();                                                                                    \
    std::size_t _doctest_subcase_idx = 0;                                                   \
    for (; _doctest_subcase_idx < _doctest_subcase_count; _doctest_subcase_idx++) {         \
        DOCTEST_SUBCASE(_doctest_subcases[_doctest_subcase_idx].c_str()) {                  \
            data = static_cast<decltype(data)>((from) + _doctest_subcase_idx);              \
        }                                                                                   \
    }                                                                                       \
    DOCTEST_CAPTURE(data);
#endif

using storage_type = std::int64_t;
using u_storage_type = std::uint64_t;
constexpr storage_type storage_type_min = std::numeric_limits<storage_type>::min();
constexpr storage_type storage_type_max = std::numeric_limits<storage_type>::max();
constexpr u_storage_type u_storage_type_min = std::numeric_limits<u_storage_type>::min();
constexpr u_storage_type u_storage_type_max = std::numeric_limits<u_storage_type>::max();

template <typename T>
using type_decay_basic_t = std::remove_cv_t<std::remove_pointer_t<std::remove_cv_t<std::remove_reference_t<T>>>>;

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
using type_decay_t = type_decay_basic_t<type_decay_ptr_t<type_decay_basic_t<std::remove_cv_t<T>>>>;

class expression;
class unary_expression;
class binary_expression;
class add_expression;
class sub_expression;
class mul_expression;
class div_expression;
class minus_expression;
class atom_expression;
class integral_expression;
class number_expression;
class zero_expression;
class variable_expression;

struct format_raise_precedence_t
{
    explicit format_raise_precedence_t() = default;
};

const format_raise_precedence_t format_raise_precedence;

class expression // NOLINT(hicpp-special-member-functions, cppcoreguidelines-special-member-functions)
{
protected:
    virtual bool equal(const expression& other) const = 0;

    bool equal_types(const expression& other) const
    {
        return std::type_index(typeid(*this)) == std::type_index(typeid(other));
    }

public:
    class format_context
    {
        std::ostream* stream_;
        bool implicit_scopes_ = false;
    public:
        class scope;
    private:
        scope* current_scope_ = nullptr;
    public:
        explicit format_context(std::ostream& stream)
            : stream_(&stream)
        {
        }

        format_context& operator<<(const expression& obj)
        {
            return obj.format(*this);
        }

        format_context& operator<<(const std::shared_ptr<expression>& obj)
        {
            return obj->format(*this);
        }

        format_context& operator<<(const std::shared_ptr<const expression>& obj)
        {
            return obj->format(*this);
        }

        /**
         * Should we assume there is defined operator precedence and there is no need to explicitly define the order of operations?
         * true - omit parentheses when not necessary
         * false - always print all parentheses
         */
        bool implicit_scopes() const
        {
            return implicit_scopes_;
        }

        void set_implicit_scopes(const bool implicit_scopes)
        {
            implicit_scopes_ = implicit_scopes;
        }

        scope* current_scope() const
        {
            return current_scope_;
        }

        format_context& operator<<(const format_raise_precedence_t&)
        {
            current_scope()->set_precedence(current_scope()->precedence() - 1);
            return *this;
        }

        // OK, I was too lazy to write all the variants. Sorry for this SFINAE mess.
        // Could be a bit prettier with C++17 (std::..._v<...> for type_traits).
        // At least std::..._t<...> variants are present.
        template <typename T, std::enable_if_t<!std::is_base_of<expression, type_decay_t<T>>::value>* = nullptr>
        format_context& operator<<(const T& obj)
        {
            *stream_ << obj;
            return *this;
        }

        class scope final
        {
            format_context* const context_;
            scope* const parent_;
            std::int16_t precedence_;
            bool const wrap_in_brackets_;
        public:
            scope(format_context& context, std::int16_t precedence)
                : context_(&context), parent_(context.current_scope()), precedence_(precedence),
                  wrap_in_brackets_(!context.implicit_scopes() || (parent_ ? parent_->precedence_ < precedence : false))
            {
                context.current_scope_ = this;

                if (wrap_in_brackets_)
                {
                    context << '(';
                }
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

            scope* parent() const
            {
                return parent_;
            }

            std::int16_t precedence() const
            {
                return precedence_;
            }

            void set_precedence(const std::int16_t precedence)
            {
                precedence_ = precedence;
            }

            bool wrap_in_brackets() const
            {
                return wrap_in_brackets_;
            }

            scope(const scope& other) = delete;
            scope(scope&& other) noexcept = delete;
            scope& operator=(const scope& other) = delete;
            scope& operator=(scope&& other) noexcept = delete;
        };
    };

    virtual ~expression() = default;
    virtual std::shared_ptr<expression> differentiate(const variable_expression& variable) const = 0;
    virtual format_context& format(format_context& context) const = 0;

    bool operator==(const expression& other) const
    {
        return equal(other);
    }
};

bool operator==(const std::shared_ptr<const expression>& lhs, const std::shared_ptr<const expression>& rhs)
{
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

class unary_expression : public expression
{
    std::shared_ptr<expression> inner_;

public:
    unary_expression(const std::shared_ptr<expression>& inner)
        : inner_(inner)
    {
        assert(inner);
    }

    std::shared_ptr<expression> inner() const
    {
        return inner_;
    }

protected:
    bool equal(const expression& other) const override
    {
        if (!equal_types(other)) return false;

        const auto* const p_other = dynamic_cast<const unary_expression*>(&other);
        return inner() == p_other->inner();
    }
};

class binary_expression : public expression
{
    std::shared_ptr<expression> left_;
    std::shared_ptr<expression> right_;

public:
    binary_expression(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
        : left_(left),
          right_(right)
    {
        assert(left && right);
    }

    std::shared_ptr<expression> left() const
    {
        return left_;
    }

    std::shared_ptr<expression> right() const
    {
        return right_;
    }

protected:
    bool equal(const expression& other) const override
    {
        if (!equal_types(other)) return false;

        const auto* const p_other = dynamic_cast<const binary_expression*>(&other);
        return left() == p_other->left() && right() == p_other->right();
    }
};

// A bit of SFINAE again. I hate copy-paste, and prefer to avoid preprocessor macros.
template <typename T, std::enable_if_t<std::is_base_of<unary_expression, type_decay_t<T>>::value>* = nullptr>
std::shared_ptr<expression> create(const std::shared_ptr<expression>& inner)
{
    if (!inner)
        return nullptr;

    return std::make_shared<T>(inner);
}

template <typename T, std::enable_if_t<std::is_base_of<binary_expression, type_decay_t<T>>::value>* = nullptr>
std::shared_ptr<expression> create(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
{
    if (!left && !right)
        return nullptr;

    if (left && !right)
        return left;

    if (!left && right)
        return right;

    return std::make_shared<T>(left, right);
}

class minus_expression final : public unary_expression
{
public:
    minus_expression(const std::shared_ptr<expression>& inner)
        : unary_expression(inner)
    {
    }

    ~minus_expression() override = default;

    minus_expression(const minus_expression& other) = default;
    minus_expression(minus_expression&& other) noexcept = default;
    minus_expression& operator=(const minus_expression& other) = default;
    minus_expression& operator=(minus_expression&& other) noexcept = default;

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override
    {
        return create<minus_expression>(inner()->differentiate(variable));
    }

    format_context& format(format_context& context) const override
    {
        format_context::scope scope(context, 3);
        return context << '-' << format_raise_precedence << inner();
    }
};

class add_expression final : public binary_expression
{
public:
    add_expression(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
        : binary_expression(left, right)
    {
    }

    ~add_expression() override = default;

    add_expression(const add_expression& other) = default;
    add_expression(add_expression&& other) noexcept = default;
    add_expression& operator=(const add_expression& other) = default;
    add_expression& operator=(add_expression&& other) noexcept = default;

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override
    {
        return create<add_expression>(left()->differentiate(variable), right()->differentiate(variable));
    }

    format_context& format(format_context& context) const override
    {
        format_context::scope scope(context, 6);
        return context << left() << '+' << right();
    }
};

class sub_expression final : public binary_expression
{
public:
    sub_expression(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
        : binary_expression(left, right)
    {
    }

    ~sub_expression() override = default;

    sub_expression(const sub_expression& other) = default;
    sub_expression(sub_expression&& other) noexcept = default;
    sub_expression& operator=(const sub_expression& other) = default;
    sub_expression& operator=(sub_expression&& other) noexcept = default;

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override
    {
        return create<sub_expression>(left()->differentiate(variable), right()->differentiate(variable));
    }

    format_context& format(format_context& context) const override
    {
        format_context::scope scope(context, 6);
        return context << left() << '-' << format_raise_precedence << right();
    }
};

class mul_expression final : public binary_expression
{
public:
    mul_expression(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
        : binary_expression(left, right)
    {
    }

    ~mul_expression() override = default;

    mul_expression(const mul_expression& other) = default;
    mul_expression(mul_expression&& other) noexcept = default;
    mul_expression& operator=(const mul_expression& other) = default;
    mul_expression& operator=(mul_expression&& other) noexcept = default;

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override
    {
        auto const lhs = create<mul_expression>(left()->differentiate(variable), right());
        auto const rhs = create<mul_expression>(left(), right()->differentiate(variable));
        return create<add_expression>(lhs, rhs);
    }

    format_context& format(format_context& context) const override
    {
        format_context::scope scope(context, 5);
        return context << left() << '*' << right();
    }
};

class div_expression final : public binary_expression
{
public:
    div_expression(const std::shared_ptr<expression>& left, const std::shared_ptr<expression>& right)
        : binary_expression(left, right)
    {
    }

    ~div_expression() override = default;

    div_expression(const div_expression& other) = default;
    div_expression(div_expression&& other) noexcept = default;
    div_expression& operator=(const div_expression& other) = default;
    div_expression& operator=(div_expression&& other) noexcept = default;

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override
    {
        auto const lhs = create<mul_expression>(left()->differentiate(variable), right());
        auto const rhs = create<mul_expression>(left(), right()->differentiate(variable));
        auto const num = create<sub_expression>(lhs, rhs);
        auto const den = create<mul_expression>(right(), right());
        return create<div_expression>(num, den);
    }

    format_context& format(format_context& context) const override
    {
        format_context::scope scope(context, 5);
        return context << left() << '/' << format_raise_precedence << right();
    }
};

class atom_expression : public expression
{
};

class variable_expression final : public atom_expression
{
    std::string name_;
public:
    explicit variable_expression(std::string name)
        : name_(std::move(name))
    {
    }

    ~variable_expression() override = default;

    variable_expression(const variable_expression& other) = default;
    variable_expression(variable_expression&& other) noexcept = default;
    variable_expression& operator=(const variable_expression& other) = default;
    variable_expression& operator=(variable_expression&& other) noexcept = default;

    std::string name() const
    {
        return name_;
    }

    std::shared_ptr<expression> differentiate(const variable_expression& variable) const override;

    format_context& format(format_context& context) const override
    {
        return context << name();
    }

protected:
    bool equal(const expression& other) const override
    {
        if (!equal_types(other)) return false;

        const auto* const p_other = dynamic_cast<const variable_expression*>(&other);
        return name() == p_other->name();
    }
};

class integral_expression : public atom_expression
{
public:
    virtual storage_type value() const = 0;

    std::shared_ptr<expression> differentiate(const variable_expression&) const override;

    format_context& format(format_context& context) const override
    {
        return context << value();
    }

protected:
    bool equal(const expression& other) const override
    {
        const auto* const p_other = dynamic_cast<const integral_expression*>(&other);
        return value() == p_other->value();
    }
};

class number_expression final : public integral_expression
{
    storage_type value_;

public:
    explicit number_expression(const storage_type value)
        : value_(value)
    {
    }

    ~number_expression() override = default;

    number_expression(const number_expression& other) = default;
    number_expression(number_expression&& other) noexcept = default;
    number_expression& operator=(const number_expression& other) = default;
    number_expression& operator=(number_expression&& other) noexcept = default;

    storage_type value() const override
    {
        return value_;
    }
};

class zero_expression final : public integral_expression
{
public:
    zero_expression() = default;

    ~zero_expression() override = default;

    zero_expression(const zero_expression& other) = default;
    zero_expression(zero_expression&& other) noexcept = default;
    zero_expression& operator=(const zero_expression& other) = default;
    zero_expression& operator=(zero_expression&& other) noexcept = default;

    storage_type value() const override
    {
        return 0;
    }
};

std::shared_ptr<integral_expression> create(const storage_type value)
{
    if (!value)
        return std::make_shared<zero_expression>();

    return std::make_shared<number_expression>(value);
}

std::shared_ptr<expression> variable_expression::differentiate(const variable_expression& variable) const
{
    if (this->name() == variable.name())
        return create(1);

    return create(0);
}

std::shared_ptr<expression> integral_expression::differentiate(const variable_expression&) const
{
    return create(0);
}

/// Microsoft STL std::from_chars (C++17) implementation (C++14 backport)
///
/// Dropped support for bases other than 10, types other than storage_type.
/// Simplified the code due to these changes.

struct from_chars_result
{
    const char* ptr;
    std::errc ec;
};

inline unsigned char digit_from_char(const char ch) noexcept
{
    if (ch < '0' || ch > '9')
        return 255u;
    return static_cast<unsigned char>(ch - '0');
}

from_chars_result integer_from_chars(
    const char* const first, const char* const last, storage_type& raw_value) noexcept
{
    auto minus_sign = false;

    const auto* next = first;

    if (next != last && *next == '-')
    {
        minus_sign = true;
        ++next;
    }

    u_storage_type risky_val;
    u_storage_type max_digit;

    if (minus_sign)
    {
        const u_storage_type abs_int_min = static_cast<u_storage_type>(storage_type_max) + 1u;

        risky_val = static_cast<u_storage_type>(abs_int_min / 10);
        max_digit = static_cast<u_storage_type>(abs_int_min % 10);
    }
    else
    {
        risky_val = static_cast<u_storage_type>(storage_type_max / 10);
        max_digit = static_cast<u_storage_type>(storage_type_max % 10);
    }

    u_storage_type value = 0;

    auto overflowed = false;

    for (; next != last; ++next)
    {
        const unsigned char digit = digit_from_char(*next);

        if (digit >= 10)
        {
            break;
        }

        if (value < risky_val // never overflows
            || (value == risky_val && digit <= max_digit))
        {
            // overflows for certain digits
            value = static_cast<u_storage_type>(value * 10 + digit);
        }
        else
        {
            // _Value > _Risky_val always overflows
            overflowed = true; // keep going, _Next still needs to be updated, _Value is now irrelevant
        }
    }

    if (next - first == static_cast<std::ptrdiff_t>(minus_sign))
    {
        return {first, std::errc::invalid_argument};
    }

    if (overflowed)
    {
        return {next, std::errc::result_out_of_range};
    }

    if (minus_sign)
    {
        value = static_cast<u_storage_type>(0 - value);
    }

    raw_value = static_cast<storage_type>(value); // implementation-defined for negative, N4713 7.8 [conv.integral]/3

    return {next, std::errc{}};
}

/// End of Microsoft STL std::from_chars implementation

enum class operator_kind
{
    unknown,
    add,
    sub,
    mul,
    div,
    pow,
    minus,
};

struct rpn_stack_variant_bracket_open_t
{
    explicit rpn_stack_variant_bracket_open_t() = default;
};

const rpn_stack_variant_bracket_open_t rpn_stack_variant_bracket_open;

// Now. The following 2 structures should really be std::variant
// But std::variant is VS2017+ only, while the minimum requirement is VS2015.
//
// So, horrible as it is, this is a necessary sacrifice.

struct rpn_stack_variant final
{
    enum { stack_operator, stack_open } kind;

    union
    {
        ::operator_kind operator_kind = {};
    };

    explicit rpn_stack_variant(const ::operator_kind operator_kind)
        : kind(stack_operator), operator_kind(operator_kind)
    {
    }

    explicit rpn_stack_variant(const rpn_stack_variant_bracket_open_t)
        : kind(stack_open)
    {
    }

    ~rpn_stack_variant() = default;

    rpn_stack_variant(const rpn_stack_variant& other) = default;
    rpn_stack_variant(rpn_stack_variant&& other) noexcept = default;
    rpn_stack_variant& operator=(const rpn_stack_variant& other) = default;
    rpn_stack_variant& operator=(rpn_stack_variant&& other) noexcept = default;
};

struct rpn_postfix_variant final
{
    enum { kind_operator, kind_number, kind_variable } kind;

    // Union is more reasonable here, but causes default destructor, copy/move constructors/operator= to be deleted
    // due to class.copy.assign#7.1
    ::operator_kind operator_kind = {};
    storage_type value = storage_type{};
    std::string variable_name;

    explicit rpn_postfix_variant(const ::operator_kind operator_kind)
        : kind(kind_operator), operator_kind(operator_kind)
    {
    }

    explicit rpn_postfix_variant(const storage_type value)
        : kind(kind_number), value(value)
    {
    }

    explicit rpn_postfix_variant(std::string variable_name)
        : kind(kind_variable), variable_name(std::move(variable_name))
    {
    }

    ~rpn_postfix_variant() = default;

    rpn_postfix_variant(const rpn_postfix_variant& other) = default;
    rpn_postfix_variant(rpn_postfix_variant&& other) noexcept = default;
    rpn_postfix_variant& operator=(const rpn_postfix_variant& other) = default;
    rpn_postfix_variant& operator=(rpn_postfix_variant&& other) noexcept = default;
};

bool parse_is_variable_char(const char c)
{
    return std::isalnum(c) || c == '_';
}

bool parse_is_left_associative(const operator_kind kind)
{
    switch (kind)
    {
        case operator_kind::pow:
        case operator_kind::minus:
            return false;
        default:
            return true;
    }
}

bool parse_check_precedence(const operator_kind& new_operator_kind, const operator_kind& top_operator_kind)
{
    if (top_operator_kind > new_operator_kind)
        return true;

    return top_operator_kind == new_operator_kind && parse_is_left_associative(top_operator_kind);
}

enum class parse_last_state
{
    none,
    stack,
    value,
};

void parse_shunting_yard(const std::string& line, std::vector<rpn_postfix_variant>& output)
{
    std::stack<rpn_stack_variant> stack;

    const auto* first = line.c_str();
    const auto* const last = line.c_str() + line.length();

    auto last_state = parse_last_state::none;

    while (first < last)
    {
        while (first < last && std::isspace(*first))
        {
            ++first;
        }

        if (first >= last)
            continue;

        // Unary minus protection:
        // (1+-5)   | Stack | Unary
        // (-5)     | Stack | Unary
        // ((1)-5)  | Value | Binary
        // -1       | None  | Unary
        if (first[0] != '-' || last_state != parse_last_state::value)
        {
            storage_type number_value;
            const auto number_parse_result = integer_from_chars(first, last, number_value);

            if (number_parse_result.ec == std::errc{})
            {
                output.emplace_back(number_value);
                last_state = parse_last_state::value;
                first = number_parse_result.ptr;
                continue;
            }
        }

        auto operator_kind = operator_kind::unknown;

        // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
        switch (first[0])
        {
            case '+':
                operator_kind = operator_kind::add;
                break;
            case '-':
                operator_kind = last_state == parse_last_state::value ? operator_kind::sub : operator_kind::minus;
                break;
            case '*':
                operator_kind = operator_kind::mul;
                break;
            case '/':
                operator_kind = operator_kind::div;
                break;
            case '^':
                operator_kind = operator_kind::pow;
                break;
        }

        if (operator_kind != operator_kind::unknown)
        {
            while (!stack.empty() && stack.top().kind == rpn_stack_variant::stack_operator)
            {
                const auto& top = stack.top();

                if (!parse_check_precedence(operator_kind, top.operator_kind))
                {
                    break;
                }

                output.emplace_back(top.operator_kind);
                stack.pop();
            }

            stack.emplace(operator_kind);
            last_state = parse_last_state::stack;

            ++first;
            continue;
        }

        // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
        switch (first[0])
        {
            case '(':
                stack.emplace(rpn_stack_variant_bracket_open);
                last_state = parse_last_state::stack;
                ++first;
                continue;

            case ')':

                while (!stack.empty() && stack.top().kind != rpn_stack_variant::stack_open)
                {
                    output.emplace_back(stack.top().operator_kind);
                    stack.pop();
                }

                if (stack.empty() || stack.top().kind != rpn_stack_variant::stack_open)
                {
                    throw std::length_error("Invalid parentheses sequence: no opening bracket for a closing one");
                }

                stack.pop();
                last_state = parse_last_state::value;
                ++first;
                continue;
        }

        if (!parse_is_variable_char(*first))
        {
            throw std::out_of_range(std::string("Unexpected character in input: '") + *first + "'");
        }

        const auto* start = first;

        while (first < last && parse_is_variable_char(*first))
        {
            ++first;
        }

        output.emplace_back(std::string(start, first - start));
        last_state = parse_last_state::value;
    }

    while (!stack.empty())
    {
        output.emplace_back(stack.top().operator_kind);
        stack.pop();
    }
}

void parse_extract_args(const std::size_t count, std::stack<std::shared_ptr<expression>>& stack,
                        std::deque<std::shared_ptr<expression>>& arguments)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        arguments.emplace_front(stack.top());
        stack.pop();
    }
}

std::shared_ptr<expression> parse(const std::string& line)
{
    std::vector<rpn_postfix_variant> output;
    parse_shunting_yard(line, output);

    std::stack<std::shared_ptr<expression>> stack;

    for (const auto& item : output)
    {
        switch (item.kind)
        {
            case rpn_postfix_variant::kind_operator:
            {
                std::deque<std::shared_ptr<expression>> arguments;
                switch (item.operator_kind)
                {
                    case operator_kind::unknown:
                        throw std::domain_error("A token has been added to RPN as an unknown operator");
                    case operator_kind::add:
                    case operator_kind::sub:
                    case operator_kind::mul:
                    case operator_kind::div:
                    case operator_kind::pow:
                        parse_extract_args(2, stack, arguments);
                        break;
                    case operator_kind::minus:
                        parse_extract_args(1, stack, arguments);
                        break;
                    default:
                        throw std::domain_error("Operator has been parsed to RPN, but not recognized at expression tree build time");
                }

                switch (item.operator_kind)
                {
                    case operator_kind::add:
                        stack.emplace(create<add_expression>(arguments.front(), arguments.back()));
                        break;
                    case operator_kind::sub:
                        stack.emplace(create<sub_expression>(arguments.front(), arguments.back()));
                        break;
                    case operator_kind::mul:
                        stack.emplace(create<mul_expression>(arguments.front(), arguments.back()));
                        break;
                    case operator_kind::div:
                        stack.emplace(create<div_expression>(arguments.front(), arguments.back()));
                        break;
                    case operator_kind::pow:
                        throw std::domain_error("Power operator not implemented");
                    case operator_kind::minus:
                        stack.emplace(create<minus_expression>(arguments.front()));
                        break;
                    default:
                        throw std::domain_error("Operator has recognized its arguments, but there is no known way to create corresponding expression");
                }
                break;
            }
            case rpn_postfix_variant::kind_number:
                stack.emplace(new number_expression(item.value));
                break;
            case rpn_postfix_variant::kind_variable:
                stack.emplace(new variable_expression(item.variable_name));
                break;
        }
    }

    if (stack.size() != 1)
    {
        const std::string message = "Invalid expression stack: expected only one element in stack after expression tree creation, got ";
        throw std::length_error(message + std::to_string(stack.size()));
    }

    return stack.top();
}

std::istream& operator>>(std::istream& in, std::shared_ptr<expression>& expression)
{
    std::string line;
    std::getline(in, line);
    expression = parse(line);
    return in;
}

variable_expression variable_x("x");

// ReSharper disable once CppParameterMayBeConst
int main(int argc, char** argv) // NOLINT(bugprone-exception-escape)
{
#ifdef DOCTEST_CONFIG_IMPLEMENT
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto res = context.run();
    if (context.shouldExit())
    {
        return res;
    }
#endif

    std::ifstream fin("input.txt");

    if (!fin.is_open())
    {
        return 0;
    }

    std::ofstream fout("output.txt");

    std::shared_ptr<expression> root;
    fin >> root;

    auto const derivative = root->differentiate(variable_x);

    expression::format_context fmt_context(fout);
    fmt_context.set_implicit_scopes(false);
    derivative->format(fmt_context);

    return 0;
}

#ifdef DOCTEST_CONFIG_IMPLEMENT
std::string str(std::shared_ptr<const expression> const& expression, bool implicit_scopes = false)
{
    std::ostringstream buffer;

    expression::format_context context(buffer);
    context.set_implicit_scopes(implicit_scopes);
    expression->format(context);

    return buffer.str();
}

TEST_CASE("numbers are printed correctly")
{
    std::int64_t n = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(n, -50, 50);

    CHECK_EQ(str(create(n)), std::to_string(n));
}

TEST_CASE("numbers are parsed correctly")
{
    from_chars_result r;
    storage_type x;

    // std::from_chars testcases are adapted from LLVM libc++ charconv test suite
    {
        const char s[] = "001x";

        // the expected form of the subject sequence is a sequence of
        // letters and digits representing an integer with the radix
        // specified by base (C11 7.22.1.4/3)
        r = integer_from_chars(s, s + sizeof(s), x);
        CHECK_EQ(r.ec, std::errc{});
        CHECK_EQ(r.ptr, s + 3);
        CHECK_EQ(x, 1);
    }

    {
        // If the pattern allows for an optional sign,
        // but the string has no digit characters following the sign,
        const char s[] = "- 9+12";
        r = integer_from_chars(s, s + sizeof(s), x);
        // no characters match the pattern.
        CHECK_EQ(r.ptr, s);
        CHECK_EQ(r.ec, std::errc::invalid_argument);
    }

    {
        const char s[] = "9+12";
        r = integer_from_chars(s, s + sizeof(s), x);
        CHECK_EQ(r.ec, std::errc{});
        // The member ptr of the return value points to the first character
        // not matching the pattern,
        CHECK_EQ(r.ptr, s + 1);
        CHECK_EQ(x, 9);
    }

    {
        const char s[] = "12";
        r = integer_from_chars(s, s + 2, x);
        CHECK_EQ(r.ec, std::errc{});
        // or has the value last if all characters match.
        CHECK_EQ(r.ptr, s + 2);
        CHECK_EQ(x, 12);
    }

    {
        // '-' is the only sign that may appear
        const char s[] = "+30";
        // If no characters match the pattern,
        r = integer_from_chars(s, s + sizeof(s), x);
        // value is unmodified,
        CHECK_EQ(x, 12);
        // the member ptr of the return value is first and
        CHECK_EQ(r.ptr, s);
        // the member ec is equal to errc::invalid_argument.
        CHECK_EQ(r.ec, std::errc::invalid_argument);
    }
}

std::shared_ptr<integral_expression> test_one = create(1);
std::shared_ptr<integral_expression> test_two = create(2);
std::shared_ptr<integral_expression> test_three = create(3);
std::shared_ptr<expression> test_one_plus_two = create<add_expression>(test_one, test_two);
std::shared_ptr<expression> test_three_plus_two = create<add_expression>(test_three, test_two);
variable_expression variable_y("y");
variable_expression variable_z("z");

void test_1_plus_2(std::shared_ptr<const expression> const& expression)
{
    // Check using equality operator
    CHECK(*test_one_plus_two == *expression);
    CHECK(*test_three_plus_two != *expression);
}

TEST_CASE("parser is working")
{
    SUBCASE("1 + 2")
    {
        test_1_plus_2(parse("1 + 2"));
    }
    SUBCASE("1+2")
    {
        test_1_plus_2(parse("1+2"));
    }
    SUBCASE("  1 +2 \\n  ")
    {
        test_1_plus_2(parse("  1 +2 \n  "));
    }
}

TEST_CASE("formatting is working")
{
    SUBCASE("1 + 2")
    {
        auto const expression = parse("1 + 2");

        CHECK_EQ(str(expression), "(1+2)");
        CHECK_EQ(str(expression, true), "1+2");
    }
    SUBCASE("((1+2)-2)")
    {
        std::ostringstream buffer;

        auto const expression = parse("((1+2)-2)");

        CHECK_EQ(str(expression), "((1+2)-2)");
        CHECK_EQ(str(expression, true), "1+2-2");
    }
    SUBCASE("2*x")
    {
        auto const expression = parse("2*x");

        CHECK_EQ(str(expression), "(2*x)");
        CHECK_EQ(str(expression, true), "2*x");
    }
    SUBCASE("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3")
    {
        CHECK_THROWS_WITH_AS(parse("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3"), "Power operator not implemented", std::domain_error);
    }
    SUBCASE("1-(2-3)")
    {
        std::ostringstream buffer;

        auto const expression = parse("1-(2-3)");

        CHECK_EQ(str(expression, true), "1-(2-3)");
    }
    SUBCASE("1-(2+3)")
    {
        std::ostringstream buffer;

        auto const expression = parse("1-(2+3)");

        CHECK_EQ(str(expression, true), "1-(2+3)");
    }
    SUBCASE("1+(2-3)")
    {
        std::ostringstream buffer;

        auto const expression = parse("1+(2-3)");

        CHECK_EQ(str(expression, true), "1+2-3");
    }
    SUBCASE("1+(2+3)")
    {
        std::ostringstream buffer;

        auto const expression = parse("1+(2+3)");

        CHECK_EQ(str(expression, true), "1+2+3");
    }
    SUBCASE("-(5+1)")
    {
        std::ostringstream buffer;

        auto const expression = parse("-(5+1)");

        CHECK_EQ(str(expression, true), "-(5+1)");
    }
}

TEST_CASE("complete suite")
{
    SUBCASE("sample")
    {
        auto const expression = parse("(((2*x)+(x*x))-3)");

        CHECK_EQ(str(expression), "(((2*x)+(x*x))-3)");
        CHECK_EQ(str(expression, true), "2*x+x*x-3");

        auto const derivative = expression->differentiate(variable_x);

        CHECK_EQ(str(derivative), "((((0*x)+(2*1))+((1*x)+(x*1)))-0)");
        CHECK_EQ(str(derivative, true), "0*x+2*1+1*x+x*1-0");
    }

    SUBCASE("analysis 1")
    {
        auto const expression = parse("(x*x*x*x+8*x*y*y*y)/(x+2*y)");

        CHECK_EQ(str(expression, true), "(x*x*x*x+8*x*y*y*y)/(x+2*y)");

        auto const derivative_x = expression->differentiate(variable_x);

        CHECK_EQ(str(derivative_x, true), "3*x*x-4*x*y+4*y*y");

        auto const derivative_xx = derivative_x->differentiate(variable_x);

        CHECK_EQ(str(derivative_xx, true), "6*x-4*y");

        auto const derivative_xxy = derivative_x->differentiate(variable_y);

        CHECK_EQ(str(derivative_xxy, true), "-4");
    }

    SUBCASE("analysis 2")
    {
        auto const expression = parse("x*y*y*z*z*z");

        CHECK_EQ(str(expression, true), "x*y*y*z*z*z");

        auto const derivative_z = expression->differentiate(variable_z);

        CHECK_EQ(str(derivative_z, true), "3*x*y*y*z*z");

        auto const derivative_zy = derivative_z->differentiate(variable_y);

        CHECK_EQ(str(derivative_zy, true), "6*x*y*z*z");

        auto const derivative_zyx = derivative_zy->differentiate(variable_x);

        CHECK_EQ(str(derivative_zyx, true), "6*y*z*z");
    }
}

// This one I was too lazy to write myself. CC-BY-SA 4.0 at https://stackoverflow.com/a/50631844
// I use this one only for static_assert's to check implementation consistency
template <typename L, typename R>
struct has_operator_equals_impl
{
    template <typename T = L, typename U = R> // template parameters here to enable SFINAE
    static auto test(T&& t, U&& u) -> decltype(t == u, void(), std::true_type{});
    static auto test(...) -> std::false_type;
    using type = decltype(test(std::declval<L>(), std::declval<R>()));
};

// Those are my own though.
template <typename L, typename R>
struct has_operator_equals
{
private:
    using type_forward = typename has_operator_equals_impl<L, R>::type;
    using type_backward = typename has_operator_equals_impl<R, L>::type;
public:
    typedef std::conditional_t<type_forward::value && type_backward::value, std::true_type, std::false_type> type;
};

template <typename L, typename R>
using has_operator_equals_t = typename has_operator_equals<L, R>::type;

static_assert(has_operator_equals_t<number_expression, expression>::value, "number_expression");
static_assert(has_operator_equals_t<variable_expression, expression>::value, "VariableExpression");
static_assert(has_operator_equals_t<add_expression, expression>::value, "AddExpression");
static_assert(has_operator_equals_t<sub_expression, expression>::value, "SubExpression");
static_assert(has_operator_equals_t<mul_expression, expression>::value, "MulExpression");
static_assert(has_operator_equals_t<integral_expression, expression>::value, "IntegralExpression");
static_assert(has_operator_equals_t<number_expression, expression>::value, "number_expression");
static_assert(has_operator_equals_t<zero_expression, expression>::value, "ZeroExpression");
#endif
