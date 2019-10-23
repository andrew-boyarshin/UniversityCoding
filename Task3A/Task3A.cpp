#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cassert>
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

        // Destructor is intentionally deleted. Change the semantics?
        // delete ptr_;
    }

public:
    // 22.2.4.2, constructors, copy, and assignment
    hash_map_node_handle() noexcept = default;

    explicit hash_map_node_handle(NodeType* const ptr) noexcept
        : ptr_(ptr)
    {
    }

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

    [[nodiscard]] bucket_type& by_index(const size_type index)
    {
        if (index >= used)
        {
            throw std::out_of_range("by_index: out of range");
        }

        return array[index];
    }

    [[nodiscard]] const bucket_type& by_index(const size_type index) const
    {
        if (index >= used)
        {
            throw std::out_of_range("by_index: out of range");
        }

        return array[index];
    }

    [[nodiscard]] bucket_type& by_hash(const size_type hash)
    {
        return by_index(bucket_index_by_hash(hash, used));
    }

    [[nodiscard]] const bucket_type& by_hash(const size_type hash) const
    {
        return by_index(bucket_index_by_hash(hash, used));
    }

private:
    template <typename, typename, typename, typename>
    friend struct hash_map;
};

struct hash_map_iterator_end_t
{
    explicit hash_map_iterator_end_t() = default;
};

const hash_map_iterator_end_t hash_map_iterator_end;

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
struct hash_map;

template <typename BucketType, bool Const, bool Local>
struct hash_map_flexible_iterator
{
private:
    using detail_value_type = typename BucketType::item_value_type;
    using detail_node_type = typename BucketType::node_type;
    using detail_self = hash_map_flexible_iterator<BucketType, Const, Local>;
    using detail_self_non_const = hash_map_flexible_iterator<BucketType, false, Local>;
    using detail_self_const = hash_map_flexible_iterator<BucketType, true, Local>;
    using detail_self_local = hash_map_flexible_iterator<BucketType, Const, true>;
    using detail_self_non_local = hash_map_flexible_iterator<BucketType, Const, false>;

public:
    using iterator_category = std::forward_iterator_tag;

    using bucket_type = const BucketType;
    using value_type = std::conditional_t<Const, const detail_value_type, detail_value_type>;
    using node_type = std::conditional_t<Const, const detail_node_type, detail_node_type>;

    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

private:
    node_type* node_ = nullptr;
    bucket_type* bucket_ = nullptr;

    template<bool Condition = Local>
    std::enable_if_t<!Condition> roll_to_node() noexcept
    {
        while (bucket_ && !node_)
        {
            bucket_ = bucket_->next_bucket;
            node_ = bucket_ ? bucket_->head : nullptr;
        }
    }

    template<bool Condition = Local>
    std::enable_if_t<Condition> roll_to_node() noexcept
    {
    }

    void verify_node() const
    {
        if (node_)
        {
            return;
        }

        throw std::out_of_range("Attempt to dereference a non-dereferenceable iterator");
    }

    template <bool T = Local, std::enable_if_t<T>* = nullptr>
    explicit hash_map_flexible_iterator(bucket_type* const bucket, const hash_map_iterator_end_t&)
        : node_(nullptr), bucket_(bucket)
    {
        if (!bucket)
        {
            throw std::exception("hash_map_flexible_iterator: local iterator must be attached to a bucket");
        }
    }

    template <bool T = Local, std::enable_if_t<T>* = nullptr>
    explicit hash_map_flexible_iterator(bucket_type& bucket, const hash_map_iterator_end_t&)
        : hash_map_flexible_iterator(std::addressof(bucket), hash_map_iterator_end)
    {
    }

    template <bool T = Local, std::enable_if_t<!T>* = nullptr>
    explicit hash_map_flexible_iterator(const hash_map_iterator_end_t&) noexcept
        : node_(nullptr), bucket_(nullptr)
    {
    }

    explicit hash_map_flexible_iterator(bucket_type* const bucket, node_type* const node = nullptr)
        : node_(node), bucket_(bucket)
    {
        if constexpr (Local)
        {
            if (!bucket)
            {
                throw std::exception("hash_map_flexible_iterator: !bucket");
            }
        }
        else
        {
            if (!bucket_ && node_)
            {
                throw std::exception("hash_map_flexible_iterator: !bucket && node");
            }

            if (!bucket_ && !node_)
            {
                throw std::exception("hash_map_flexible_iterator: !bucket && !node");
            }
        }

        node_ = bucket_->head;
        roll_to_node();
    }

    explicit hash_map_flexible_iterator(bucket_type& bucket, node_type* const node = nullptr)
        : hash_map_flexible_iterator(std::addressof(bucket), node)
    {
    }

    template <bool T = Const, std::enable_if_t<T>* = nullptr>
    explicit hash_map_flexible_iterator(const detail_self_non_const& other)
        : hash_map_flexible_iterator(other.bucket_, other.node_)
    {
    }

    template <bool T = Local, std::enable_if_t<!T>* = nullptr>
    explicit hash_map_flexible_iterator(const detail_self_local& other)
        : hash_map_flexible_iterator(other.bucket_, other.node_)
    {
    }

    template <typename, typename, typename, typename>
    friend struct hash_map;

    friend detail_self_non_const;
    friend detail_self_const;
    friend detail_self_non_local;
    friend detail_self_local;

public:
    hash_map_flexible_iterator() = delete;
    ~hash_map_flexible_iterator() = default;
    hash_map_flexible_iterator(const hash_map_flexible_iterator& other) = default;
    hash_map_flexible_iterator(hash_map_flexible_iterator&& other) noexcept = default;
    hash_map_flexible_iterator& operator=(const hash_map_flexible_iterator& other) = default;
    hash_map_flexible_iterator& operator=(hash_map_flexible_iterator&& other) noexcept = default;

    detail_self& operator++() noexcept
    {
        if (node_)
        {
            node_ = node_->next;
            roll_to_node();
        }
        return *this;
    }

    detail_self operator++(int)
    {
        const detail_self copy(*this);
        ++(*this);
        return copy;
    }

    reference operator*() const
    {
        verify_node();

        return node_->get();
    }

    pointer operator->() const
    {
        verify_node();

        return std::addressof(node_->get());
    }

    friend bool operator==(const detail_self& lhs, const detail_self& rhs) noexcept
    {
        return lhs.node_ == rhs.node_ && lhs.bucket_ == rhs.bucket_;
    }

    friend bool operator!=(const detail_self& lhs, const detail_self& rhs) noexcept
    {
        return !(lhs == rhs);
    }
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
    using iterator = hash_map_flexible_iterator<bucket_type, false, false>;
    using const_iterator = hash_map_flexible_iterator<bucket_type, true, false>;
    using local_iterator = hash_map_flexible_iterator<bucket_type, false, true>;
    using const_local_iterator = hash_map_flexible_iterator<bucket_type, true, true>;

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

    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

    [[nodiscard]] size_type size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] size_type max_size() const noexcept
    {
        return std::numeric_limits<difference_type>::max();
    }

    iterator begin()
    {
        if (bucket_count() == 0)
        {
            return end();
        }

        return iterator{bucket_list_.by_index(0)};
    }

    iterator end()
    {
        return iterator{hash_map_iterator_end};
    }

    [[nodiscard]] const_iterator begin() const
    {
        if (bucket_count() == 0)
        {
            return end();
        }

        return const_iterator{bucket_list_.by_index(0)};
    }

    [[nodiscard]] const_iterator end() const noexcept
    {
        return const_iterator{hash_map_iterator_end};
    }

    [[nodiscard]] const_iterator cbegin() const
    {
        if (bucket_count() == 0)
        {
            return cend();
        }

        return const_iterator{bucket_list_.by_index(0)};
    }

    [[nodiscard]] const_iterator cend() const noexcept
    {
        return const_iterator{hash_map_iterator_end};
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
        {
            return node_type();
        }

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

    iterator erase(iterator position)
    {
        return erase(const_iterator(position));
    }

    size_type erase(const key_type& k)
    {
        iterator it = find(k);
        if (it == end())
        {
            return 0;
        }

        erase(it);
        return 1;
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        for (const_iterator previous = first; first != last; previous = first)
        {
            ++first;
            erase(previous);
        }

        return iterator(last.bucket_, last.node_);
    }

    void clear() noexcept;
    template <typename H2, typename P2>
    void merge(hash_map<Key, T, H2, P2>& source);
    template <typename H2, typename P2>
    void merge(hash_map<Key, T, H2, P2>&& source);

    [[nodiscard]] hasher hash_function() const
    {
        return hasher_;
    }

    [[nodiscard]] key_equal key_eq() const
    {
        return key_equal_;
    }

    [[nodiscard]] iterator find(const key_type& k);
    [[nodiscard]] const_iterator find(const key_type& k) const;

    [[nodiscard]] size_type count(const key_type& k) const
    {
        return find(k) != end() ? 1u : 0u;
    }

    [[nodiscard]] bool contains(const key_type& k) const
    {
        return find(k) != end();
    }

    [[nodiscard]] std::pair<iterator, iterator> equal_range(const key_type& k)
    {
        iterator left = find(k), right = left;
        if (left != end())
        {
            ++right;
        }

        return std::make_pair(left, right);
    }

    [[nodiscard]] std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const
    {
        const_iterator left = find(k), right = left;
        if (left != end())
        {
            ++right;
        }

        return std::make_pair(left, right);
    }

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
        {
            throw std::out_of_range("at: key not found");
        }

        return it->second;
    }

    const mapped_type& at(const key_type& k) const
    {
        const_iterator it = find(k);
        if (it == end())
        {
            throw std::out_of_range("at: key not found");
        }

        return it->second;
    }

    // bucket interface
    [[nodiscard]] size_type bucket_count() const noexcept
    {
        return bucket_list_.used;
    }

    [[nodiscard]] size_type max_bucket_count() const noexcept
    {
        return max_size();
    }

    [[nodiscard]] size_type bucket_size(size_type n) const;

    [[nodiscard]] size_type bucket(const key_type& k) const
    {
        auto count = bucket_count();
        if (count == 0)
        {
            throw std::exception("bucket: count is 0");
        }

        return bucket_index_by_hash(hasher_(k), count);
    }

    [[nodiscard]] local_iterator begin(size_type n)
    {
        return local_iterator{bucket_list_.by_index(n)};
    }

    [[nodiscard]] local_iterator end(size_type n)
    {
        return local_iterator{bucket_list_.by_index(n), hash_map_iterator_end};
    }

    [[nodiscard]] const_local_iterator begin(size_type n) const
    {
        return const_local_iterator{bucket_list_.by_index(n)};
    }

    [[nodiscard]] const_local_iterator end(size_type n) const
    {
        return const_local_iterator{bucket_list_.by_index(n), hash_map_iterator_end};
    }

    [[nodiscard]] const_local_iterator cbegin(size_type n) const
    {
        return const_local_iterator{bucket_list_.by_index(n)};
    }

    [[nodiscard]] const_local_iterator cend(size_type n) const
    {
        return const_local_iterator{bucket_list_.by_index(n), hash_map_iterator_end};
    }

    // hash policy
    [[nodiscard]] float load_factor() const noexcept
    {
        const auto buckets = bucket_count();
        return buckets != 0 ? static_cast<float>(size()) / buckets : 0.f;
    }

    [[nodiscard]] float max_load_factor() const noexcept
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

    template <bool Const,
              typename LocalIt = hash_map_flexible_iterator<bucket_type, Const, true>,
              typename GlobalIt = hash_map_flexible_iterator<bucket_type, Const, false>>
    [[nodiscard]] GlobalIt find_within(const bucket_type& bucket, const key_type& key)
    {
        LocalIt it_end{bucket, hash_map_iterator_end};

        for (LocalIt it{bucket}; it != it_end; ++it)
        {
            if (key_equal_(it->first, key))
            {
                return GlobalIt(it);
            }
        }

        return end();
    }

    [[nodiscard]] auto find_within(const bucket_type& bucket, const key_type& key)
    {
        return find_within<false>(bucket, key);
    }

    [[nodiscard]] auto find_within(const bucket_type& bucket, const key_type& key) const
    {
        return find_within<true>(bucket, key);
    }

    template<typename... Args>
    std::pair<iterator, bool> emplace_key_args(const key_type& k, Args&&... args);

    node_type remove(const_iterator it);
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
typename hash_map<Key, T, Hash, Equal>::node_type hash_map<Key, T, Hash, Equal>::extract(const_iterator position)
{
    return remove(position);
}

template <typename Key, typename T, typename Hash, typename Equal>
void hash_map<Key, T, Hash, Equal>::clear() noexcept
{
    const iterator it_end = end();
    for (iterator it = begin(); it != it_end;)
    {
        it = erase(it);
    }

    DOCTEST_CHECK_EQ(size(), 0);
}

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::iterator hash_map<Key, T, Hash, Equal>::erase(const_iterator position)
{
    iterator next(position.bucket_, const_cast<typename iterator::node_type*>(position.node_));
    ++next;
    remove(position);
    return next;
}

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::iterator hash_map<Key, T, Hash, Equal>::find(const key_type& k)
{
    const auto hash = hasher_(k);
    auto bucket_count = this->bucket_count();

    if (bucket_count == 0)
    {
        return end();
    }

    auto& bucket = bucket_list_.by_hash(hash);

    return find_within(bucket, k);
}

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::const_iterator hash_map<Key, T, Hash, Equal>::find(const key_type& k) const
{
    const auto hash = hasher_(k);
    auto bucket_count = this->bucket_count();

    if (bucket_count == 0)
    {
        return end();
    }

    auto& bucket = bucket_list_.by_hash(hash);

    return find_within(bucket, k);
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
        typename iterator::node_type* node = it.node_;

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

template <typename Key, typename T, typename Hash, typename Equal>
typename hash_map<Key, T, Hash, Equal>::node_type hash_map<Key, T, Hash, Equal>::remove(const_iterator it)
{
    const auto key = it->first;
    const auto bucket_index = bucket(key);

    local_iterator bucket_it = begin(bucket_index);
    const local_iterator bucket_it_end = end(bucket_index);

    DOCTEST_CHECK_NE(size_--, 0);

    if (key_equal_(key, bucket_it->first))
    {
        bucket_list_.by_index(bucket_index).head = bucket_it.node_->next;
        return node_type{ bucket_it.node_ };
    }

    for (; bucket_it != bucket_it_end; )
    {
        local_iterator previous = bucket_it++;

        if (key_equal_(key, bucket_it->first))
        {
            previous.node_->next = bucket_it.node_->next;
            return node_type{ bucket_it.node_ };
        }
    }

    throw std::exception("remove: failed to find element in bucket");
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

    CHECK_EQ(map.size(), 3);

    map.clear();
    CHECK_EQ(map.size(), 0);
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
