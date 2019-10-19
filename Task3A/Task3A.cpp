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

    next_pointer next = nullptr;

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

template <typename T>
struct hash_map_bucket final
{
    using node_type = T;
    using item_value_type = typename node_type::value_type;
    using self_type = hash_map_bucket<node_type>;
    using next_pointer = self_type*;

#ifdef DOCTEST_CONFIG_IMPLEMENT
    static_assert(!std::is_pointer_v<node_type>, "hash_map_bucket<T*> not allowed");
    static_assert(std::is_same_v<T*, typename node_type::next_pointer>, "node pointer type mismatch");
#endif

    node_type* head = nullptr;
    next_pointer next_bucket = nullptr;

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

    bucket_type& by_index(const size_type index)
    {
        return array[index];
    }

    const bucket_type& by_index(const size_type index) const
    {
        return array[index];
    }

    bucket_type& by_hash(const size_type hash)
    {
        return by_index(bucket_index_by_hash(hash, used));
    }

    const bucket_type& by_hash(const size_type hash) const
    {
        return by_index(bucket_index_by_hash(hash, used));
    }

private:
    template <typename, typename, typename, typename>
    friend struct hash_map;
};

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
struct hash_map;

template <typename QualifiedNodeType, typename BucketType, typename Derived>
struct hash_map_iterator_base
{
protected:
    using derived = Derived;

    using bucket_type = BucketType;
    using value_type = typename bucket_type::item_value_type;
    using qualified_node_type = QualifiedNodeType;

    static_assert(std::is_same_v<typename bucket_type::node_type, std::remove_const_t<QualifiedNodeType>>, "Node type mismatch (using const-aware comparison)");

    qualified_node_type* node_ = nullptr;
    const bucket_type* bucket_ = nullptr;

    void verify_node() const
    {
        if (node_)
        {
            return;
        }

        throw std::out_of_range("Attempt to dereference a non-dereferenceable iterator");
    }

    void roll_to_node()
    {
        while (bucket_ && !node_)
        {
            bucket_ = bucket_->next_bucket;
            node_ = bucket_ ? bucket_->head : nullptr;
        }
    }

    void initialize(const bucket_type* const bucket, qualified_node_type* const node)
    {
        bucket_ = bucket;
        node_ = node;

        if (!bucket_)
        {
            if (node_)
            {
                throw std::exception("iterator::initialize: !bucket && node");
            }

            return;
        }

        node_ = bucket_->head;
        roll_to_node();
    }

    template <typename, typename, typename, typename>
    friend struct hash_map;

public:
    derived& operator++()
    {
        if (node_)
        {
            node_ = node_->next;
            roll_to_node();
        }
        return static_cast<derived&>(*this);
    }

    derived& operator++(int)
    {
        derived copy(*this);
        ++(*this);
        return copy;
    }

    friend bool operator==(const derived& lhs, const derived& rhs)
    {
        return lhs.node_ == rhs.node_ && lhs.bucket_ == rhs.bucket_;
    }

    friend bool operator!=(const derived& lhs, const derived& rhs)
    {
        return !(lhs == rhs);
    }
};

template <typename BucketType>
struct hash_map_iterator final : hash_map_iterator_base<typename BucketType::node_type, BucketType, hash_map_iterator<BucketType>>
{
    using iterator_category = std::forward_iterator_tag;

    using base = hash_map_iterator_base<typename BucketType::node_type, BucketType, hash_map_iterator<BucketType>>;

    using bucket_type = typename base::bucket_type;
    using value_type = typename base::value_type;
    using qualified_node_type = typename base::qualified_node_type;

    hash_map_iterator() = delete;
    ~hash_map_iterator() = default;
    hash_map_iterator(const hash_map_iterator& other) = default;
    hash_map_iterator(hash_map_iterator&& other) noexcept = default;
    hash_map_iterator& operator=(const hash_map_iterator& other) = default;
    hash_map_iterator& operator=(hash_map_iterator&& other) noexcept = default;

    value_type& operator*() const
    {
        verify_node();

        return node_->get();
    }

    value_type* operator->() const
    {
        verify_node();

        return std::addressof(node_->get());
    }

private:
    explicit hash_map_iterator(bucket_type* const bucket, qualified_node_type* const node = nullptr)
    {
        initialize(bucket, node);
    }

    explicit hash_map_iterator(bucket_type& bucket, qualified_node_type* const node = nullptr)
        : hash_map_iterator(std::addressof(bucket), node)
    {
    }

    template <typename, typename, typename, typename>
    friend struct hash_map;
};

template <typename BucketType>
struct hash_map_const_iterator final : hash_map_iterator_base<const typename BucketType::node_type, BucketType, hash_map_const_iterator<BucketType>>
{
    using iterator_category = std::forward_iterator_tag;

    using base = hash_map_iterator_base<const typename BucketType::node_type, BucketType, hash_map_const_iterator<BucketType>>;

    using bucket_type = typename base::bucket_type;
    using value_type = typename base::value_type;
    using qualified_node_type = typename base::qualified_node_type;

    hash_map_const_iterator() = delete;
    ~hash_map_const_iterator() = default;
    hash_map_const_iterator(const hash_map_const_iterator& other) = default;
    hash_map_const_iterator(hash_map_const_iterator&& other) noexcept = default;
    hash_map_const_iterator& operator=(const hash_map_const_iterator& other) = default;
    hash_map_const_iterator& operator=(hash_map_const_iterator&& other) noexcept = default;

    const value_type& operator*() const
    {
        verify_node();

        return node_->get();
    }

    const value_type* operator->() const
    {
        verify_node();

        return std::addressof(node_->get());
    }

private:
    explicit hash_map_const_iterator(const bucket_type* const bucket, qualified_node_type* const node = nullptr)
    {
        initialize(bucket, node);
    }

    explicit hash_map_const_iterator(const bucket_type& bucket, qualified_node_type* const node = nullptr)
        : hash_map_const_iterator(std::addressof(bucket), node)
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
    using iterator = hash_map_iterator<bucket_type>;
    using const_iterator = hash_map_const_iterator<bucket_type>;
    using local_iterator = hash_map_iterator<bucket_type>;
    using const_local_iterator = hash_map_iterator<bucket_type>;

private:
    using bucket_list_type = hash_map_bucket_list<bucket_type, size_type>;
    using is_nothrow_1 = std::is_nothrow_default_constructible<hasher>;
    using is_nothrow_2 = std::is_nothrow_default_constructible<key_equal>;

    constexpr static size_type default_bucket_count = 64;

    bucket_list_type bucket_list_;
    hasher hasher_;
    key_equal key_equal_;
    size_type size_ = 0;
    float max_load_factor_ = 1.0f;

public:
    hash_map() noexcept(is_nothrow_1::value && is_nothrow_2::value)
        : hash_map(default_bucket_count)
    {
    }

    explicit hash_map(size_type n, const hasher& hf = hasher(), const key_equal& eql = key_equal());

    template<typename InputIterator>
    hash_map(InputIterator f, InputIterator l, size_type n = default_bucket_count, const hasher& hf = hasher(), const key_equal& eql = key_equal());

    hash_map(std::initializer_list<value_type> il, size_type n = default_bucket_count, const hasher& hf = hasher(), const key_equal& eql = key_equal());

    bool empty() const noexcept
    {
        return size() == 0;
    }

    size_type size() const noexcept
    {
        return size_;
    }

    size_type max_size() const noexcept
    {
        return std::numeric_limits<difference_type>::max();
    }

    iterator begin() noexcept
    {
        return iterator{bucket_list_.by_index(0)};
    }

    iterator end() noexcept
    {
        return iterator{nullptr};
    }

    const_iterator begin() const noexcept
    {
        return const_iterator{bucket_list_.by_index(0)};
    }

    const_iterator end() const noexcept
    {
        return const_iterator{nullptr};
    }

    const_iterator cbegin() const noexcept
    {
        return const_iterator{bucket_list_.by_index(0)};
    }

    const_iterator cend() const noexcept
    {
        return const_iterator{nullptr};
    }

    template <typename... Args>
    iterator emplace(Args&&... args);
    template <typename... Args>
    iterator emplace_hint(const_iterator position, Args&&... args);
    iterator insert(const value_type& obj);

    template <typename P>
    iterator insert(P&& obj)
    {
        return emplace(std::forward<P>(obj));
    }

    iterator insert(const_iterator, const value_type& obj)
    {
        return insert(obj).first;
    }

    template <typename P>
    iterator insert(const_iterator hint, P&& obj)
    {
        return emplace_hint(hint, std::forward<P>(obj));
    }

    template <typename InputIterator>
    void insert(InputIterator first, InputIterator last);

    void insert(std::initializer_list<value_type> list)
    {
        insert(list.begin(), list.end());
    }

    node_type extract(const_iterator position); // C++17

    node_type extract(const key_type& x)
    {
        iterator it = find(x);
        if (it == end())
            return node_type();
        return extract(it);
    }

    // insert_return_type insert(node_type&& nh); // C++17
    iterator insert(const_iterator hint, node_type&& nh); // C++17

    template<typename... Args>
    std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args)
    {
        return emplace_key_args(k, std::piecewise_construct, std::forward_as_tuple(k), std::forward_as_tuple(std::forward<Args>(args)...));
    }

    template<typename... Args>
    std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args)
    {
        return emplace_key_args(k, std::piecewise_construct, std::forward_as_tuple(std::move(k)), std::forward_as_tuple(std::forward<Args>(args)...));
    }

    template<typename... Args>
    iterator try_emplace(const_iterator, const key_type& k, Args&&... args)
    {
        return try_emplace(k, std::forward<Args>(args)...).first;
    }

    template<typename... Args>
    iterator try_emplace(const_iterator, key_type&& k, Args&&... args)
    {
        return try_emplace(std::move(k), std::forward<Args>(args)...).first;
    }

    template<typename M>
    std::pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj);
    template<typename M>
    std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj);

    template<typename M>
    iterator insert_or_assign(const_iterator, const key_type& k, M&& obj)
    {
        return insert_or_assign(k, std::forward<M>(obj)).first;
    }

    template<typename M>
    iterator insert_or_assign(const_iterator, key_type&& k, M&& obj)
    {
        return insert_or_assign(std::move(k), std::forward<M>(obj)).first;
    }


    iterator erase(const_iterator position);
    iterator erase(iterator position); // C++14
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
        return try_emplace(std::move(k)).first->second;
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

    float max_load_factor() const noexcept
    {
        return max_load_factor_;
    }

    void max_load_factor(float z)
    {
        max_load_factor_ = std::max(z, load_factor());
    }

    void rehash(size_type n)
    {
        if (n < 2)
        {
            n = 2;
        }

        const auto bucket_count = this->bucket_count();
        if (n > bucket_count)
        {
            realloc(n);
            return;
        }

        // do sth clever with n < bucket_count
        realloc(n);
    }

    void reserve(size_type n);

private:
    void realloc(size_type n);

    iterator find_within(bucket_type& bucket, const key_type& key)
    {
        bucket_node_type* it = bucket.head;

        for (; it; it = it->next)
        {
            if (key_equal_(it->get().first, key))
            {
                return iterator{bucket, it};
            }
        }

        return end();
    }

    const_iterator find_within(const bucket_type& bucket, const key_type& key) const
    {
        bucket_node_type* it = bucket.head;

        for (; it; it = it->next)
        {
            if (key_equal_(it->get().first, key))
            {
                return const_iterator{bucket, it};
            }
        }

        return end();
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace_key_args(const key_type& k, Args&&... args);
};

template <typename Key, typename T, typename Hash, typename Equal>
hash_map<Key, T, Hash, Equal>::hash_map(const size_type n, const hasher& hf, const key_equal& eql)
    : hasher_(hf), key_equal_(eql)
{
    this->realloc(n);
}

template <typename Key, typename T, typename Hash, typename Equal>
template <typename InputIterator>
void hash_map<Key, T, Hash, Equal>::insert(InputIterator first, InputIterator last)
{
    for (; first != last; ++first)
    {
        this->insert(*first);
    }
}

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::size_type hash_map<Key, T, Hash, Equal>::bucket_size(const size_type n) const
{
    auto& bucket = bucket_list_.by_index(n);

    size_type count = 0;

    const auto* it = bucket.head;
    for (; it; it = it->next)
    {
        count++;
    }

    return count;
}

template <typename Key, typename T, typename Hash, typename Equal>
void hash_map<Key, T, Hash, Equal>::realloc(const size_type n)
{
    auto ptr = std::make_unique<bucket_type[]>(n);

    for (size_type i = 0; i < n - 1; ++i)
    {
        ptr[i].next_bucket = &ptr[i + 1];
    }

    iterator it = begin();
    const iterator it_end = end();
    for (; it != it_end;)
    {
        typename iterator::qualified_node_type* node = it.node_;

        ++it;

        const auto hash = hasher_(node->get().first);
        const auto index = bucket_index_by_hash(hash, n);

        node->next = ptr[index].head;
        ptr[index].head = node;
    }

    bucket_list_.array = std::move(ptr);
    bucket_list_.used = n;
}

template <typename Key, typename T, typename Hash, typename Equal>
template <typename ... Args>
std::pair<typename hash_map<Key, T, Hash, Equal>::iterator, bool> hash_map<Key, T, Hash, Equal>::emplace_key_args(
    const key_type& k, Args&&... args)
{
    const auto hash = hasher_(k);
    auto bucket_count = this->bucket_count();
    const auto bucket_pressure = bucket_count * max_load_factor();

    if (!bucket_count || size() + 1 > bucket_pressure)
    {
        rehash(std::max<size_type>(2 * bucket_count,
            size_type(ceil(float(size() + 1) / max_load_factor()))));

        bucket_count = this->bucket_count();
    }

    auto& bucket = bucket_list_.by_hash(hash);
    
    iterator entry = find_within(bucket, k);

    if (entry != end())
    {
        return std::make_pair(entry, false);
    }

    auto* node = new bucket_node_type(typename bucket_node_type::value_type{ std::forward<Args>(args)... });

    node->next = bucket.head;
    bucket.head = node;

    size_++;

    return std::make_pair(iterator{ bucket, node }, true);
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

TEST_CASE("map is created correctly")
{
    std::int64_t n = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(n, 1, 50);

    hash_map<std::uint64_t, std::uint8_t> map(n);
    CHECK_EQ(map.bucket_count(), n);
    CHECK_EQ(map.size(), 0);
    CHECK_EQ(map.empty(), true);
    CHECK_EQ(map.begin(), map.end());
    CHECK_EQ(map.cbegin(), map.cend());
}

using my_pair = std::pair<const std::uint64_t, std::uint8_t>;
bool operator==(const my_pair& lhs, const my_pair& rhs)
{
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

TEST_CASE("map elements are emplaced correctly")
{
    hash_map<std::uint64_t, std::uint8_t> map;
    CHECK_NE(map.bucket_count(), 0);
    CHECK_EQ(map.size(), 0);
    CHECK_EQ(map.empty(), true);
    CHECK_EQ(map.begin(), map.end());
    CHECK_EQ(map.cbegin(), map.cend());

    map.try_emplace(20, 1);

    CHECK_EQ(map.size(), 1);
    CHECK_EQ(map.empty(), false);
    CHECK_NE(map.begin(), map.end());
    CHECK_NE(map.cbegin(), map.cend());

    map.try_emplace(40, 50);

    CHECK_EQ(map.size(), 2);
    CHECK_EQ(map.empty(), false);
    CHECK_NE(map.begin(), map.end());
    CHECK_NE(map.cbegin(), map.cend());

    std::size_t count = 0, bits = 0;

    for (auto && pair : map)
    {
        count++;
        switch (pair.first)
        {
        case 20:
            CHECK_EQ(bits & 1, 0);
            bits |= 1;
            CHECK_EQ(pair.second, 1);
            break;
        case 40:
            CHECK_EQ(bits & 2, 0);
            bits |= 2;
            CHECK_EQ(pair.second, 50);
            break;
        default:
            DOCTEST_FAIL_CHECK("nodefault");
        }
    }

    CHECK_EQ(count, 2);
    CHECK_EQ(bits, 3);

    CHECK_EQ(map[20], 1);
    map[20] = 5;
    CHECK_EQ(map[20], 5);
    CHECK_EQ(map[40], 50);
    CHECK_EQ(map[60], 0);
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
