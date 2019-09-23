#include <iostream>
#include <fstream>
#include <cstdint>
#include <random>
#include <cassert>

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
constexpr storage_type storage_type_min = std::numeric_limits<storage_type>::min();
constexpr storage_type storage_type_max = std::numeric_limits<storage_type>::max();
using default_random_engine = std::mt19937_64;

struct random_fill_t
{
    explicit random_fill_t() = default;
};

const random_fill_t random_fill;

struct default_fill_t
{
    explicit default_fill_t() = default;
};

const default_fill_t default_fill;

struct zero_fill_t
{
    explicit zero_fill_t() = default;
};

const zero_fill_t zero_fill;

// cppcoreguidelines check is stupid, it thinks this class contains non-default constructor
// it even continues to think so when "<...> = default;" constructor is removed

struct diagonal_fill_t // NOLINT(cppcoreguidelines-special-member-functions)
{
    diagonal_fill_t() = default;
    virtual ~diagonal_fill_t() = default;

    virtual storage_type operator[](std::size_t index) const noexcept = 0;
};

struct diagonal_fill_with final : diagonal_fill_t
{
    storage_type value;

    explicit diagonal_fill_with(const storage_type& value) : value(value)
    {
    }

    ~diagonal_fill_with() override = default;

    diagonal_fill_with(const diagonal_fill_with& other) = default;
    diagonal_fill_with(diagonal_fill_with&& other) noexcept = default;
    diagonal_fill_with& operator=(const diagonal_fill_with& other) = default;
    diagonal_fill_with& operator=(diagonal_fill_with&& other) noexcept = default;

    storage_type operator[](const std::size_t) const noexcept override
    {
        return value;
    }
};

struct diagonal_fill_from final : diagonal_fill_t
{
    const storage_type* value;

    explicit diagonal_fill_from(const storage_type (&value)[]) : value(value)
    {
        // Note: diagonal_fill_from is unsafe (no bounds checking)
        // It can be made much safer using compile-time calculated sizes
        // But that requires some templating, which is absolutely & unfortunately not part of the first task
    }

    ~diagonal_fill_from() override = default;

    diagonal_fill_from(const diagonal_fill_from& other) = default;
    diagonal_fill_from(diagonal_fill_from&& other) noexcept = default;
    diagonal_fill_from& operator=(const diagonal_fill_from& other) = default;
    diagonal_fill_from& operator=(diagonal_fill_from&& other) noexcept = default;

    storage_type operator[](const std::size_t index) const noexcept override
    {
        return value[index];
    }
};

struct complete_fill_with final
{
    storage_type value;

    explicit complete_fill_with(const storage_type& value) : value(value)
    {
    }

    ~complete_fill_with() = default;

    complete_fill_with(const complete_fill_with& other) = default;
    complete_fill_with(complete_fill_with&& other) noexcept = default;
    complete_fill_with& operator=(const complete_fill_with& other) = default;
    complete_fill_with& operator=(complete_fill_with&& other) noexcept = default;
};

// Note: template class
// Note: custom allocators
class Matrix final
{
    // Note: using storage_type = <...>
    // Note: using size_type = <...>

    std::size_t size_ = 0;
    storage_type** storage_ = nullptr;

    explicit Matrix(const std::size_t n) : size_(n), storage_(new storage_type* [n])
    {
        assert(n >= 1);
    }

public:
    Matrix(std::size_t n, const default_fill_t&) : Matrix(n)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            this->storage_[i] = new storage_type[n];
        }
    }

    Matrix(std::size_t n, const zero_fill_t&) : Matrix(n)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            this->storage_[i] = new storage_type[n]();
        }
    }

    Matrix(std::size_t n, const random_fill_t&) : Matrix(n, default_fill)
    {
        std::random_device random_seed;
        default_random_engine random(random_seed());
        const std::uniform_int_distribution<storage_type> distribution(storage_type_min, storage_type_max);

        for (std::size_t i = 0; i < n; i++)
        {
            for (std::size_t j = 0; j < n; j++)
            {
                this->storage_[i][j] = distribution(random);
            }
        }
    }

    Matrix(std::size_t n, const diagonal_fill_t& value) : Matrix(n, zero_fill)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            this->storage_[i][i] = value[i];
        }
    }

    Matrix(std::size_t n, const complete_fill_with& value) : Matrix(n, default_fill)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            for (std::size_t j = 0; j < n; j++)
            {
                this->storage_[i][j] = value.value;
            }
        }
    }

    Matrix(const Matrix& other) : Matrix(other.size(), default_fill)
    {
        for (std::size_t i = 0; i < size(); i++)
        {
            std::memcpy(this->storage_[i], other.storage_[i], size() * sizeof(storage_type));
        }
    }

    Matrix(Matrix&& other) noexcept : size_(other.size_),
                                      storage_(std::exchange(other.storage_, nullptr))
    {
    }

    ~Matrix()
    {
        if (this->storage_ == nullptr)
        {
            return;
        }

        for (std::size_t i = 0; i < size(); i++)
        {
            delete[] this->storage_[i];
        }

        delete[] this->storage_;
    }

    std::size_t size() const
    {
        assert(this->storage_ != nullptr);

        return this->size_;
    }

    storage_type& operator()(const std::size_t row, const std::size_t column)
    {
        assert(this->storage_ != nullptr);
        assert(row < size());
        assert(column < size());

        return this->storage_[row][column];
    }

    const storage_type& operator()(const std::size_t row, const std::size_t column) const
    {
        assert(this->storage_ != nullptr);
        assert(row < size());
        assert(column < size());

        return this->storage_[row][column];
    }

    class RowProxy final
    {
        const Matrix* container_ = nullptr;
        storage_type* row_ = nullptr;

        friend Matrix;

        RowProxy(const Matrix* const container, storage_type* const row)
            : container_(container), row_(row)
        {
        }

    public:
        ~RowProxy() = default;

        RowProxy(const RowProxy& other) = default;
        RowProxy(RowProxy&& other) noexcept = default;
        RowProxy& operator=(const RowProxy& other) = default;
        RowProxy& operator=(RowProxy&& other) noexcept = default;

        std::size_t size() const
        {
            assert(this->container_ != nullptr);

            return this->container_->size();
        }

        storage_type& operator[](const std::size_t column)
        {
            assert(this->row_ != nullptr);
            assert(column < size());

            return this->row_[column];
        }

        const storage_type& operator[](const std::size_t column) const
        {
            assert(this->row_ != nullptr);
            assert(column < size());

            return this->row_[column];
        }
    };

    RowProxy operator[](const std::size_t row)
    {
        assert(this->storage_ != nullptr);
        assert(row < size());

        return {this, this->storage_[row]};
    }

    RowProxy operator[](const std::size_t row) const
    {
        assert(this->storage_ != nullptr);
        assert(row < size());

        return { this, this->storage_[row] };
    }

    void swap(Matrix& that) noexcept
    {
        using std::swap;
        swap(this->size_, that.size_);
        swap(this->storage_, that.storage_);
    }

    Matrix& operator=(Matrix other) noexcept
    {
        swap(other);
        return *this;
    }

    Matrix& operator+=(const Matrix& rhs)
    {
        assert(this->size() == rhs.size());

        for (std::size_t i = 0; i < size(); i++)
        {
            for (std::size_t j = 0; j < size(); j++)
            {
                this->storage_[i][j] += rhs.storage_[i][j];
            }
        }

        return *this;
    }

    Matrix& transpose()
    {
        using std::swap;

        for (std::size_t i = 0; i < size(); i++)
        {
            for (std::size_t j = 0; j < i; j++)
            {
                swap(this->storage_[i][j], this->storage_[j][i]);
            }
        }

        return *this;
    }

    Matrix operator-() const
    {
        const auto n = size();

        Matrix result(n, default_fill);

        for (std::size_t i = 0; i < n; i++)
        {
            for (std::size_t j = 0; j < n; j++)
            {
                result(i, j) = -this->storage_[i][j];
            }
        }

        return result;
    }
};

Matrix operator+(Matrix lhs, const Matrix& rhs)
{
    return lhs += rhs;
}

Matrix operator*(const Matrix& lhs, const Matrix& rhs)
{
    assert(lhs.size() == rhs.size());

    const auto size = lhs.size();

    Matrix result(size, default_fill);

    for (std::size_t i = 0; i < size; i++)
    {
        for (std::size_t j = 0; j < size; j++)
        {
            auto sum = storage_type();

            for (std::size_t k = 0; k < size; k++)
            {
                sum += lhs(i, k) * rhs(k, j);
            }

            result(i, j) = sum;
        }
    }

    return result;
}

Matrix transposed(Matrix lhs)
{
    return lhs.transpose();
}

std::istream& operator>>(std::istream& in, Matrix& matrix)
{
    const auto n = matrix.size();

    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            in >> matrix(i, j);
        }
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, Matrix& matrix)
{
    const auto n = matrix.size();

    for (std::size_t i = 0; i < n; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            if (j != 0)
            {
                out << ' ';
            }

            out << matrix(i, j);
        }

        out << std::endl;
    }

    return out;
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

    std::size_t n;
    storage_type k;
    fin >> n >> k;

    Matrix A(n, default_fill), B(n, default_fill), C(n, default_fill), D(n, default_fill);
    const Matrix K(n, diagonal_fill_with(k));

    fin >> A >> B >> C >> D;

    // Matrices are mutable, so create as little copies as possible.

    A += B * C.transpose();
    A += K;

    auto result = A * D.transpose();

    fout << result;

    return 0;
}

#ifdef DOCTEST_CONFIG_IMPLEMENT
void check_element_modification(Matrix& m)
{
    m(4, 4) = INT64_MAX;

    CHECK_EQ(m.size(), 5);
    CHECK_EQ(m(4, 4), INT64_MAX);
    CHECK_EQ(m[4][4], INT64_MAX);
}

void check_manual_fill(Matrix& m)
{
    CHECK_EQ(m.size(), 5);

    for (std::size_t i = 0; i < m.size(); i++)
    {
        for (std::size_t j = 0; j < m.size(); j++)
        {
            m(i, j) = i + j;
        }
    }

    CHECK_EQ(m.size(), 5);
    CHECK_NE(m(4, 4), INT64_MAX);

    for (std::size_t i = 0; i < m.size(); i++)
    {
        auto row = m[i];
        for (std::size_t j = 0; j < m.size(); j++)
        {
            CHECK_EQ(m(i, j), i + j);
            CHECK_EQ(row[j], i + j);
        }
    }
}

TEST_CASE("matrices can be created with default fill")
{
    Matrix m(5, default_fill);

    REQUIRE_EQ(m.size(), 5);

    SUBCASE("modifying the element works")
    {
        check_element_modification(m);
    }

    SUBCASE("manual filling works")
    {
        check_manual_fill(m);
    }
}

TEST_CASE("matrices can be created with zero fill")
{
    Matrix m(5, zero_fill);

    REQUIRE_EQ(m.size(), 5);

    for (std::size_t i = 0; i < m.size(); i++)
    {
        const auto row = m[i];
        for (std::size_t j = 0; j < m.size(); j++)
        {
            REQUIRE_EQ(m(i, j), 0);
            REQUIRE_EQ(row[j], 0);
        }
    }

    SUBCASE("modifying the element works")
    {
        check_element_modification(m);
    }

    SUBCASE("manual filling works")
    {
        check_manual_fill(m);
    }
}

TEST_CASE("matrices can be created with random fill")
{
    Matrix m(5, random_fill);

    REQUIRE_EQ(m.size(), 5);

    SUBCASE("modifying the element works")
    {
        check_element_modification(m);
    }

    SUBCASE("manual filling works")
    {
        check_manual_fill(m);
    }
}

TEST_CASE("matrices can be copied")
{
    SUBCASE("copy-initialized matrices is identical")
    {
        Matrix m1(5, random_fill);

        REQUIRE_EQ(m1.size(), 5);

        const auto m2 = m1;

        CHECK_EQ(m1.size(), m2.size());

        for (std::size_t i = 0; i < m1.size(); i++)
        {
            const auto row1 = m1[i];
            const auto row2 = m2[i];
            for (std::size_t j = 0; j < m1.size(); j++)
            {
                CHECK_EQ(m1(i, j), m2(i, j));
                CHECK_EQ(row1[j], row2[j]);
            }
        }
    }
}

TEST_CASE("matrices are copy-assigned correctly")
{
    std::size_t n = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(n, 1, 50);

    // Ensure there is no dependency on operator= within operator= implementation,
    // either directly, or via std::swap. The test will fail with stack overflow in case this is violated.
    // Note that this guarantee must still hold when Copy & Swap idiom is used.

    Matrix a(n, default_fill);
    Matrix b(n, default_fill);
    a = b;

    CHECK_EQ(a.size(), n);
    CHECK_EQ(b.size(), n);

    for (std::size_t i = 0; i < n; i++)
    {
        const auto rowA = a[i];
        const auto rowB = b[i];
        for (std::size_t j = 0; j < n; j++)
        {
            CHECK_EQ(a(i, j), b(i, j));
            CHECK_EQ(rowA[j], rowB[j]);
        }
    }
}

TEST_CASE("matrices can be negated")
{
    SUBCASE("negated matrices is correct")
    {
        Matrix m1(5, random_fill);

        REQUIRE_EQ(m1.size(), 5);

        const auto m2 = -m1;

        CHECK_EQ(m1.size(), m2.size());

        for (std::size_t i = 0; i < m1.size(); i++)
        {
            const auto row1 = m1[i];
            const auto row2 = m2[i];
            for (std::size_t j = 0; j < m1.size(); j++)
            {
                CHECK_EQ(m1(i, j), -m2(i, j));
                CHECK_EQ(-m1(i, j), m2(i, j));
                CHECK_EQ(row1[j], -row2[j]);
                CHECK_EQ(-row1[j], row2[j]);
            }
        }
    }
}

TEST_CASE("matrices can be added")
{
    Matrix a(5, random_fill);
    Matrix b(5, random_fill);

    REQUIRE_EQ(a.size(), 5);
    REQUIRE_EQ(b.size(), 5);

    SUBCASE("matrices addition works")
    {
        auto c = a + b;

        CHECK_EQ(a.size(), c.size());
        CHECK_EQ(b.size(), c.size());

        for (std::size_t i = 0; i < a.size(); i++)
        {
            const auto rowA = a[i];
            const auto rowB = b[i];
            const auto rowC = c[i];
            for (std::size_t j = 0; j < a.size(); j++)
            {
                CHECK_EQ(c(i, j), a(i, j) + b(i, j));
                CHECK_EQ(rowC[j], rowA[j] + rowB[j]);
            }
        }
    }

    SUBCASE("matrices negated addition works")
    {
        auto c = a + -a;

        CHECK_EQ(a.size(), c.size());

        for (std::size_t i = 0; i < a.size(); i++)
        {
            const auto rowC = c[i];
            for (std::size_t j = 0; j < a.size(); j++)
            {
                CHECK_EQ(rowC[j], 0);
            }
        }
    }
}

TEST_CASE("matrices can be multiplied")
{
    SUBCASE("matrices multiplication produces matrices of the same size")
    {
        Matrix a(5, random_fill);
        Matrix b(5, random_fill);

        CHECK_EQ(a.size(), 5);
        CHECK_EQ(b.size(), 5);

        auto c = a * b;

        CHECK_EQ(a.size(), c.size());
        CHECK_EQ(b.size(), c.size());
    }
}

TEST_CASE("2x2 matrices multiplication works")
{
    int i = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(i, 1, 50);

    Matrix a(2, random_fill);
    Matrix b(2, random_fill);

    CHECK_EQ(a.size(), 2);
    CHECK_EQ(b.size(), 2);

    auto c = a * b;

    CHECK_EQ(a.size(), c.size());
    CHECK_EQ(b.size(), c.size());

    CHECK_EQ(c(0, 0), a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0));
    CHECK_EQ(c(0, 1), a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1));
    CHECK_EQ(c(1, 0), a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0));
    CHECK_EQ(c(1, 1), a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1));
}

TEST_CASE("3x3 matrices multiplication works")
{
    int i = 0;
    DOCTEST_INDEX_PARAMETERIZED_DATA(i, 1, 50);

    Matrix a(3, random_fill);
    Matrix b(3, random_fill);

    CHECK_EQ(a.size(), 3);
    CHECK_EQ(b.size(), 3);

    auto c = a * b;

    CHECK_EQ(a.size(), c.size());
    CHECK_EQ(b.size(), c.size());

    CHECK_EQ(c(0, 0), a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0));
    CHECK_EQ(c(0, 1), a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1));
    CHECK_EQ(c(0, 2), a(0, 0) * b(0, 2) + a(0, 1) * b(1, 2) + a(0, 2) * b(2, 2));
    CHECK_EQ(c(1, 0), a(1, 0) * b(0, 0) + a(1, 1) * b(1, 0) + a(1, 2) * b(2, 0));
    CHECK_EQ(c(1, 1), a(1, 0) * b(0, 1) + a(1, 1) * b(1, 1) + a(1, 2) * b(2, 1));
    CHECK_EQ(c(1, 2), a(1, 0) * b(0, 2) + a(1, 1) * b(1, 2) + a(1, 2) * b(2, 2));
    CHECK_EQ(c(2, 0), a(2, 0) * b(0, 0) + a(2, 1) * b(1, 0) + a(2, 2) * b(2, 0));
    CHECK_EQ(c(2, 1), a(2, 0) * b(0, 1) + a(2, 1) * b(1, 1) + a(2, 2) * b(2, 1));
    CHECK_EQ(c(2, 2), a(2, 0) * b(0, 2) + a(2, 1) * b(1, 2) + a(2, 2) * b(2, 2));
}
#endif
