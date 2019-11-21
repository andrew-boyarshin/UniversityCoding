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
std::optional<std::size_t> len(std::shared_ptr<value> source);
void bind(std::shared_ptr<value> expression, std::shared_ptr<value> source);

struct value
{
    virtual ~value() = default;

    virtual std::shared_ptr<value> evaluate() = 0;

protected:
    virtual std::optional<std::partial_ordering> compare(const value* other) const
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

    void drain_generator();

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

    void update_end_state()
    {
        auto&& values = sequence->values;
        if (!end && index >= values.size())
        {
            generate_next();
        }

        end = index >= values.size();
    }

    [[nodiscard]] std::shared_ptr<value> get()
    {
        DEBUG_ASSERT(!end, assert_module{});

        return sequence->values[index];
    }

    friend struct value;
    friend struct sequence_value;
};

struct call_value final : value
{
    std::shared_ptr<value> what;
    std::shared_ptr<sequence_value> arguments;

    std::shared_ptr<value> evaluate() override;

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

    bool evaluate_condition();

    std::shared_ptr<value> evaluate() override
    {
        return evaluate_condition() ? suite_true->evaluate() : suite_false->evaluate();
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
    if (auto it = std::dynamic_pointer_cast<sequence_iterator_value>(source))
    {
        return ++*it, it;
    }

    if (auto it = std::dynamic_pointer_cast<generator_value>(source))
    {
        return it->next();
    }

    throw std::exception("Unknown next source");
}

std::optional<std::size_t> len(std::shared_ptr<value> source)
{
    if (const auto it = std::dynamic_pointer_cast<sequence_iterator_value>(source))
    {
        return len(it->sequence);
    }

    if (const auto it = std::dynamic_pointer_cast<generator_value>(source))
    {
        return len(value::create<sequence_value>(it));
    }

    if (const auto it = std::dynamic_pointer_cast<sequence_value>(source))
    {
        it->drain_generator();
        return it->values.size();
    }

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

    if (expression_variable && source_seq)
    {
        expression_variable->value = source;
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
                expression_seq->values[i] = source_seq->values[i];
            }
            return;
        }
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

void sequence_value::drain_generator()
{
    if (!generator)
        return;

    while (auto var = generator->next())
    {
        values.push_back(var);
    }
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

    if (node_name == "function")
    {
        auto&& nodes = ast->nodes;
        DEBUG_ASSERT(nodes.size() == 2, assert_module{});
        DEBUG_ASSERT(nodes[0]->name == "ident", assert_module{});
        DEBUG_ASSERT(nodes[1]->original_name == "expression", assert_module{});

        auto&& identifier = nodes[0]->token;

        const auto context = context_base::create<ast_tree_context>(ast_context);
        const auto variable = context->name_lookup->define(identifier);

        auto&& body = parse_into_ast(context, nodes[1]);

        const std::deque<sequence_value::value_type> values{ variable };
        auto&& args = value::create<sequence_value>(values);

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

    // not found
    auto var = (*ast_context->name_lookup)[node_name]->bind(ast_context->name_lookup).bound_variable;
    if (var)
    {
        auto&& nodes = ast->nodes;

        std::deque<sequence_value::value_type> values;

        for (auto && ast_argument : nodes)
        {
            values.push_back(parse_into_ast(ast_context, ast_argument));
        }

        auto&& args = value::create<sequence_value>(values);

        return value::create<call_value>(var, args);
    }

    DEBUG_UNREACHABLE(assert_module{});

    return nullptr;
}

std::shared_ptr<value> evaluate(const std::shared_ptr<value>& ast)
{
    return ast->evaluate();
}

std::optional<int64_t> evaluate_to_number(const std::shared_ptr<value>& ast)
{
    const auto result = evaluate(ast);

    if (auto num = std::dynamic_pointer_cast<integer_value>(result))
    {
        return num->value;
    }

    return std::nullopt;
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

    auto ast = peg_parser(source);

    fout << "(val 0)" << std::endl;

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

bool condition_value::evaluate_condition()
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

void test_suite()
{
    using namespace boost::ut;

    "examples"_test = [] {
        "1"_test = [] {
            auto ast = peg_parser(R"((let K = (val 10) in
                                              (add
                                                  (val 5)
                                                  (var K))))");

            expect(evaluate_to_number(ast).value() == 15_ll);
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

            expect(evaluate_to_number(ast).value() == 31_ll);
        };
        "3"_test = [] {
            auto ast = peg_parser(R"((let F = (function arg (add (var arg) (val 1))) in
                                              (let V = (val -1) in
                                                  (call (var F) (var V)))))");

            expect(evaluate_to_number(ast).value() == 0_ll);
        };
        "4"_test = [] {
            auto ast = peg_parser(R"((add (var A) (var B)))");

            expect(throws<std::exception>([&ast]() { evaluate(ast); }));
        };
    };
}
