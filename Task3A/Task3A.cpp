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
#include <bitset>

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

template <typename T>
using compat_remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename NodeType>
class hash_map_node_handle final
{
    using node_type = NodeType;
    using node_pointer_type = node_type*;
public:
    using key_type = typename node_type::key_type;
    using mapped_type = typename node_type::mapped_type;
    using value_type = std::pair<const key_type, mapped_type>;
private:
    node_pointer_type ptr_;

    void destroy_node()
    {
        if (!ptr_)
        {
            return;
        }

        delete ptr_;
    }

public:
    // 22.2.4.2, constructors, copy, and assignment
    hash_map_node_handle() noexcept = default;
    hash_map_node_handle(const hash_map_node_handle& other) = delete;
    hash_map_node_handle(hash_map_node_handle&&) noexcept = default;

    hash_map_node_handle& operator=(hash_map_node_handle&& other) noexcept
    {
        destroy_node();
        ptr_ = std::exchange(other.ptr_, nullptr);
        return *this;
    }

    hash_map_node_handle& operator=(const hash_map_node_handle& other) = delete;

    // 22.2.4.3, destructor
    ~hash_map_node_handle()
    {
        destroy_node();
    }

    // 22.2.4.4, observers
    key_type& key() const
    {
        return ptr_->ref().first;
    }

    mapped_type& mapped() const
    {
        return ptr_->ref().second;
    }

    explicit operator bool() const noexcept
    {
        return ptr_ != nullptr;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return ptr_ == nullptr;
    }

    // 22.2.4.5, modifiers
    void swap(hash_map_node_handle& other) noexcept
    {
        using std::swap;
        swap(ptr_, other.ptr_);
    }

    friend void swap(hash_map_node_handle& x, hash_map_node_handle& y) noexcept(noexcept(x.swap(y)))
    {
        x.swap(y);
    }
};

// A little quest for the reviewer: find violation of strict aliasing rule (hidden somewhere below these lines)
// and suggest a way to fix it.
//
// Want to make it tricky? Do not lose performance gains from this little trick.
// Want to make it impossible? Limit yourself by C++14 only.

template <typename Key, typename Value>
struct hash_map_node final
{
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<const key_type, mapped_type>;
    using self_type = hash_map_node<key_type, mapped_type>;
    using pointer_type = self_type*;
    using next_pointer = pointer_type;

    next_pointer* next = nullptr;

private:
    using non_const_ref_value_type = std::pair<key_type&, mapped_type&>;
    using non_const_rvalue_value_type = std::pair<key_type&&, mapped_type&&>;

    value_type storage_;

public:
    explicit hash_map_node(const value_type& storage)
        : storage_(storage)
    {
    }

    value_type& get() noexcept
    {
        return storage_;
    }

    const value_type& get() const noexcept
    {
        return storage_;
    }

private:
    non_const_ref_value_type& ref() noexcept
    {
        return std::make_pair(const_cast<key_type&>(storage_.first), storage_.second);
    }

    non_const_rvalue_value_type& move() noexcept
    {
        return std::make_pair(std::move(const_cast<key_type&>(storage_.first)), std::move(storage_.second));
    }

public:

    hash_map_node(const hash_map_node& other) = delete;
    hash_map_node(hash_map_node&& other) noexcept = delete;

    hash_map_node& operator=(const hash_map_node& other)
    {
        ref() = other.get();
        next = other.next;
        return *this;
    }

    hash_map_node& operator=(hash_map_node&& other) noexcept
    {
        ref() = other.move();
        next = std::exchange(other.next, nullptr);
        return *this;
    }

    template<typename Tuple, std::enable_if_t<std::is_same<compat_remove_cvref_t<Tuple>, value_type>::value>* = nullptr>
    hash_map_node& operator=(Tuple&& other) noexcept
    {
        // stop other from decaying into lvalue too early for std::pair constructor to catch what was passed
        ref() = std::forward<Tuple>(other);
        return *this;
    }

    ~hash_map_node() = delete;
};

template <typename Value>
struct hash_map_bucket final
{
    using value_type = Value;
    using pointer_type = typename value_type::pointer_type;
    using next_pointer = typename value_type::next_pointer;
    using self_type = hash_map_bucket<value_type>;

#ifdef DOCTEST_CONFIG_IMPLEMENT
    static_assert(!std::is_pointer_v<value_type>, "hash_map_bucket<T*> not allowed");
#endif

    next_pointer head = nullptr;
    self_type* next_bucket = nullptr;

    explicit hash_map_bucket() = default;
    ~hash_map_bucket() = default;

    hash_map_bucket(const hash_map_bucket& other) = delete;
    hash_map_bucket(hash_map_bucket&& other) noexcept = delete;
    hash_map_bucket& operator=(const hash_map_bucket& other) = delete;
    hash_map_bucket& operator=(hash_map_bucket&& other) noexcept = delete;
};

template <typename SizeType>
SizeType bucket_index_by_hash(SizeType hash, SizeType bucket_count)
{
    return hash % bucket_count;
}

template <typename BucketType, typename SizeType>
struct hash_map_bucket_list final
{
    using bucket_type = BucketType;
    using size_type = SizeType;

#ifdef DOCTEST_CONFIG_IMPLEMENT
    static_assert(!std::is_pointer_v<bucket_type>, "hash_map_bucket_list<T*> not allowed");
#endif

    std::unique_ptr<bucket_type[]> array;
    size_type used = 0;

    hash_map_bucket_list() = default;
    ~hash_map_bucket_list() = default;
    hash_map_bucket_list(const hash_map_bucket_list& other) = delete;
    hash_map_bucket_list(hash_map_bucket_list&& other) noexcept = delete;
    hash_map_bucket_list& operator=(const hash_map_bucket_list& other) = delete;
    hash_map_bucket_list& operator=(hash_map_bucket_list&& other) noexcept = delete;

    bucket_type& operator[](size_type hash)
    {
        const auto index = bucket_index_by_hash(hash, used.size());
        return array.list[index];
    }

    const bucket_type& operator[](size_type hash) const
    {
        const auto index = bucket_index_by_hash(hash, used.size());
        return array.list[index];
    }

private:
    template <typename, typename, typename, typename>
    friend struct hash_map;
};

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
struct hash_map;

template <typename Value>
struct hash_map_iterator final
{
    using iterator_category = std::forward_iterator_tag;

    using value_type = typename Value::value_type;

private:
    using next_pointer = typename Value::pointer_type;

    next_pointer node_ = nullptr;

public:
    hash_map_iterator() = default;
    ~hash_map_iterator() = default;
    hash_map_iterator(const hash_map_iterator& other) = default;
    hash_map_iterator(hash_map_iterator&& other) noexcept = default;
    hash_map_iterator& operator=(const hash_map_iterator& other) = default;
    hash_map_iterator& operator=(hash_map_iterator&& other) noexcept = default;

    value_type& operator*() const
    {
        return node_->get();
    }

    value_type* operator->() const
    {
        return std::addressof(node_->get());
    }

    hash_map_iterator& operator++()
    {
        node_ = node_->next;
        return *this;
    }

    hash_map_iterator& operator++(int)
    {
        hash_map_iterator copy(*this);
        ++(*this);
        return copy;
    }

    friend bool operator==(const hash_map_iterator& lhs, const hash_map_iterator& rhs)
    {
        return lhs.node_ == rhs.node_;
    }

    friend bool operator!=(const hash_map_iterator& lhs, const hash_map_iterator& rhs)
    {
        return !(lhs == rhs);
    }

private:
    explicit hash_map_iterator(const next_pointer& node)
        : node_(node)
    {
    }

    template <typename, typename, typename, typename>
    friend struct hash_map;
};

template <typename Key, typename T, typename Hash, typename Equal>
struct hash_map final
{
    using key_type = Key;
    using mapped_type = T;
    using hasher = Hash;
    using key_equal = Equal;

private:
    using bucket_node_type = hash_map_node<key_type, mapped_type>;
    using bucket_type = hash_map_bucket<bucket_node_type>;

public:
    using node_type = hash_map_node_handle<bucket_node_type>;
    using value_type = typename bucket_node_type::value_type;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator = hash_map_iterator<node_type>;
    using const_iterator = hash_map_iterator<node_type>;
    using local_iterator = hash_map_iterator<node_type>;
    using const_local_iterator = hash_map_iterator<node_type>;

private:
    using bucket_list_type = hash_map_bucket_list<bucket_type, size_type>;
    using is_nothrow_1 = std::is_nothrow_default_constructible<hasher>;
    using is_nothrow_2 = std::is_nothrow_default_constructible<key_equal>;

    bucket_list_type bucket_list_;
    hasher hasher_;
    key_equal key_equal_;
    size_type size_ = 0;

public:
    hash_map() noexcept(is_nothrow_1::value && is_nothrow_2::value) = default;
    bool empty() const noexcept { return size() == 0; }
    size_type size() const noexcept { return size_; }
    size_type max_size() const noexcept;
    iterator begin() noexcept;
    iterator end() noexcept
    {
        return iterator(nullptr);
    }
    const_iterator begin() const noexcept;
    const_iterator end() const noexcept
    {
        return const_iterator(nullptr);
    }
    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept
    {
        return const_iterator(nullptr);
    }
    template <class... Args>
    iterator emplace(Args&&... args);
    template <class... Args>
    iterator emplace_hint(const_iterator position, Args&&... args);
    iterator insert(const value_type& obj);

    template <class P>
    iterator insert(P&& obj)
    {
        return emplace(std::forward<P>(obj));
    }

    iterator insert(const_iterator hint, const value_type& obj);

    template <class P>
    iterator insert(const_iterator hint, P&& obj)
    {
        return emplace_hint(hint, std::forward<P>(obj));
    }

    template <class InputIterator>
    void insert(InputIterator first, InputIterator last);
    void insert(std::initializer_list<value_type>);
    node_type extract(const_iterator position); // C++17

    node_type extract(const key_type& x)
    {
        iterator it = find(x);
        if (it == end())
            return node_type();
        return extract(it);
    }

    iterator insert(node_type&& nh); // C++17
    iterator insert(const_iterator hint, node_type&& nh); // C++17
    iterator erase(const_iterator position);
    //iterator erase(iterator position); // C++14
    size_type erase(const key_type& k);
    iterator erase(const_iterator first, const_iterator last);
    void clear() noexcept;
    template <typename H2, typename P2>
    void merge(hash_map<Key, T, H2, P2>& source);
    template <typename H2, typename P2>
    void merge(hash_map<Key, T, H2, P2>&& source);

    hasher hash_function() const
    {
        return hasher_;
    }

    key_equal key_eq() const
    {
        return key_equal_;
    }

    iterator find(const key_type& k);
    const_iterator find(const key_type& k) const;
    size_type count(const key_type& k) const;

    bool contains(const key_type& k) const
    {
        return find(k) != end();
    }

    std::pair<iterator, iterator> equal_range(const key_type& k);
    std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const;

    // 22.5.4.3, element access
    mapped_type& operator[](const key_type& k)
    {
        return try_emplace(k).first->second;
    }

    mapped_type& operator[](key_type&& k)
    {
        return try_emplace(move(k)).first->second;
    }

    mapped_type& at(const key_type& k)
    {
        iterator it = find(k);
        if (it == end())
            throw std::out_of_range("at: key not found");
        return it->second;
    }

    const mapped_type& at(const key_type& k) const
    {
        const_iterator it = find(k);
        if (it == end())
            throw std::out_of_range("at: key not found");
        return it->second;
    }

    // bucket interface
    size_type bucket_count() const noexcept
    {
        return bucket_list_.used;
    }

    size_type max_bucket_count() const noexcept
    {
        return max_size();
    }

    size_type bucket_size(size_type n) const;

    size_type bucket(const key_type& k) const
    {
        auto count = bucket_count();
        if (count == 0)
        {
            throw std::exception("bucket: count is 0");
        }

        return bucket_index_by_hash(hasher(k), count);
    }

    local_iterator begin(size_type n);
    local_iterator end(size_type n);
    const_local_iterator begin(size_type n) const;
    const_local_iterator end(size_type n) const;
    const_local_iterator cbegin(size_type n) const;
    const_local_iterator cend(size_type n) const;

    // hash policy
    float load_factor() const noexcept
    {
        const auto buckets = bucket_count();
        return buckets != 0 ? static_cast<float>(size()) / buckets : 0.f;
    }

    float max_load_factor() const noexcept;
    void max_load_factor(float z);
    void rehash(size_type n);
    void reserve(size_type n);

private:
    void realloc(size_type n);
};

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::iterator hash_map<Key, T, Hash, Equal>::
begin() noexcept
{
    return iterator(bucket_list_);
}

template <typename Key, typename T, typename Hash, typename Equal>
template <class InputIterator>
void hash_map<Key, T, Hash, Equal>::insert(InputIterator first, InputIterator last)
{
    for (; first != last; ++first)
        this->insert(*first);
}

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::size_type hash_map<Key, T, Hash, Equal>::bucket_size(size_type n) const
{
    auto& bucket = bucket_list_.array[n];

    size_type count = 0;

    const auto* it = bucket.head;
    for (; it; it = it->next)
        count++;

    return count;
}

template <typename Key, typename T, typename Hash, typename Equal>
void hash_map<Key, T, Hash, Equal>::realloc(size_type n)
{
    auto ptr = std::make_unique<bucket_type[]>(n);

    auto it = begin();
    const auto it_end = end();
    for (; it != it_end; ++it)
    {
        auto* item = *it;
        const auto index = bucket_index_by_hash(hash, n);
        //ptr[index].head = 
    }

    bucket_list_.array = ptr;
    bucket_list_.used = n;
}

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

    return 0;
}

#ifdef DOCTEST_CONFIG_IMPLEMENT

TEST_CASE("numbers are printed correctly")
{
    std::int64_t n = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(n, -50, 50);

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

template< class T, class U >
constexpr bool has_assignment_operator_v = std::is_assignable<T, U>::value && !std::is_trivially_assignable<T, U>::value;

static_assert(!std::is_default_constructible_v<hash_map_node<int, int>>, "hash_map_node<K, V>() = delete");
static_assert(!std::is_copy_constructible_v<hash_map_node<int, int>>, "hash_map_node<K, V>(copy) = delete");
static_assert(std::is_copy_assignable_v<hash_map_node<int, int>>, "hash_map_node<K, V> operator=(copy)");
static_assert(!std::is_move_constructible_v<hash_map_node<int, int>>, "hash_map_node<K, V>(move) = delete");
static_assert(std::is_move_assignable_v<hash_map_node<int, int>>, "hash_map_node<K, V> operator=(move)");

static_assert(has_assignment_operator_v<hash_map_node<int, int>, std::pair<const int, int>>, "pair<const K, V>");
static_assert(has_assignment_operator_v<hash_map_node<int, int>, const std::pair<const int, int>>, "const pair<const K, V>");
static_assert(!has_assignment_operator_v<hash_map_node<int, int>, std::pair<int, int>>, "pair<K, V>");
static_assert(!has_assignment_operator_v<hash_map_node<int, int>, const std::pair<int, int>>, "const pair<K, V>");

static_assert(has_assignment_operator_v<hash_map_node<std::uint64_t, std::string>, std::pair<const std::uint64_t, std::string>>, "pair<const K, V>");
static_assert(has_assignment_operator_v<hash_map_node<std::uint64_t, std::string>, const std::pair<const std::uint64_t, std::string>>, "const pair<const K, V>");
static_assert(!has_assignment_operator_v<hash_map_node<std::uint64_t, std::string>, std::pair<std::uint64_t, std::string>>, "pair<K, V>");
static_assert(!has_assignment_operator_v<hash_map_node<std::uint64_t, std::string>, const std::pair<std::uint64_t, std::string>>, "const pair<K, V>");

#endif
