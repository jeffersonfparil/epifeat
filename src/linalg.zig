const std: type = @import("std");
const Allocator: type = std.mem.Allocator;
const expect = std.testing.expect;
const Complex = std.math.complex.Complex;

/// Error types
const MatrixError = error{
    TIsNotAFloatType,
    IncompatibleMatrices,
    NonSquareMatrix,
    NonHermitian,
    NotPositiveSemiDefinite,
    NotAColumnVector,
    NotARowVector,
    RowsLessThanColumns,
    SingularMatrix,
    NullDataInMatrix,
    ComplexNumber,
    OutOfMemory,
};

/// Float type checker of T
/// We only accept float types here because we do not wat to deal with integer overflows
/// Used only in Matrix(T).init(...) as all other Matrix instatntiation methods call init.
pub fn is_float(comptime T: type) bool {
    return (T == f16) or (T == f32) or (T == f64) or (T == f128) or (T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128));
}

/// Generic arithmetic math operations for primitive float types and complex numbers class
fn as(comptime C: type, comptime T: type, a: C) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = Complex(C).init(a, 0.0);
    } else {
        out = @as(T, a);
    }
    return out;
}

fn add(comptime T: type, a: T, b: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = a.add(b);
    } else {
        out = a + b;
    }
    return (out);
}

fn subtract(comptime T: type, a: T, b: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = a.sub(b);
    } else {
        out = a - b;
    }
    return (out);
}

fn multiply(comptime T: type, a: T, b: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = a.mul(b);
    } else {
        out = a * b;
    }
    return (out);
}

fn divide(comptime T: type, a: T, b: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = a.div(b);
    } else {
        out = a / b;
    }
    return (out);
}

fn square_root(comptime T: type, a: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = std.math.complex.sqrt(a);
    } else {
        out = @sqrt(a);
    }
    return (out);
}

fn negative(comptime T: type, a: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        out = a.neg();
    } else {
        out = -a;
    }
    return (out);
}

fn absolute(comptime T: type, a: T) T {
    var out: T = undefined;
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        const F: type = @TypeOf(a.re);
        out = Complex(F).init(@sqrt((a.re * a.re) + (a.im * a.im)), 0.0);
        // const hypotenuse = std.math.complex.abs(a);
        // return Complex(F).init(hypotenuse, 0.0);
    } else {
        out = @abs(a);
    }
    return (out);
}

fn equal_to(comptime T: type, a: T, b: T) bool {
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        const F: type = @TypeOf(a.re);
        const z_a: F = a.magnitude();
        const z_b: F = b.magnitude();
        return (z_a == z_b);
    } else {
        return (a == b);
    }
}

fn less_than(comptime T: type, a: T, b: T) bool {
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        const F: type = @TypeOf(a.re);
        const z_a: F = a.magnitude();
        const z_b: F = b.magnitude();
        return (z_a < z_b);
    } else {
        return (a < b);
    }
}

fn greater_than(comptime T: type, a: T, b: T) bool {
    if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
        const F: type = @TypeOf(a.re);
        const z_a: F = a.magnitude();
        const z_b: F = b.magnitude();
        return (z_a > z_b);
    } else {
        return (a > b);
    }
}

fn Matrix(comptime T: type) type {
    return struct {
        data: [][]T,
        n: usize,
        p: usize,
        determinant: ?T,
        const Self = @This();
        /// Initialise a matrix
        pub fn init(n: usize, p: usize, allocator: Allocator) !Self {
            if (!is_float(T)) {
                return MatrixError.TIsNotAFloatType;
            }
            const data: [][]T = try allocator.alloc([]T, n);
            for (0..n) |i| {
                data[i] = try allocator.alloc(T, p);
            }
            return .{
                .data = data,
                .n = n,
                .p = p,
                .determinant = null,
            };
        }
        /// De-initialise a matrix
        pub fn deinit(self: Self, allocator: Allocator) void {
            for (self.data) |inner| {
                allocator.free(inner);
            }
            allocator.free(self.data);
        }
        /// Initiliase an identity matrix
        pub fn init_identity(n: usize, allocator: Allocator) !Self {
            var identity = try Matrix(T).init(n, n, allocator);
            for (0..n) |i| {
                for (0..n) |j| {
                    if (i == j) {
                        identity.data[i][j] = as(f64, T, 1.0);
                    } else {
                        identity.data[i][j] = as(f64, T, 0.0);
                    }
                }
            }
            return identity;
        }
        /// Initiliase a matrix filled with a value
        pub fn init_fill(n: usize, p: usize, value: T, allocator: Allocator) !Self {
            var matrix = try Matrix(T).init(n, p, allocator);
            for (0..n) |i| {
                for (0..p) |j| {
                    matrix.data[i][j] = value;
                }
            }
            return matrix;
        }
        /// Clone a matrix
        /// We enforce that Self be constant (i.e. *cont Self, i.e. a pointer to a contant self)
        /// to make sure that self do not change values after cloning.
        pub fn clone(self: *const Self, allocator: Allocator) !Self {
            var copy = try Matrix(T).init(self.n, self.p, allocator);
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    copy.data[i][j] = self.data[i][j];
                }
            }
            return copy;
        }
        /// Clone a matrix and convert T from non-complex into complex numbers
        pub fn clone_into_complex(self: *const Self, allocator: Allocator) !Matrix(Complex(T)) {
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                return MatrixError.ComplexNumber;
            }
            var copy = try Matrix(Complex(T)).init(self.n, self.p, allocator);
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    copy.data[i][j] = Complex(T).init(self.data[i][j], 0.0);
                }
            }
            return copy;
        }
        /// Initialise a complex matrix
        pub fn init_complex(n: usize, p: usize, allocator: Allocator) !Matrix(Complex(T)) {
            const a = try Matrix(T).init(n, p, allocator);
            defer a.deinit(allocator);
            const b = try a.clone_into_complex(allocator);
            return b;
        }
        /// Initialise an identity complex matrix
        pub fn init_identity_complex(n: usize, allocator: Allocator) !Matrix(Complex(T)) {
            const a = try Matrix(T).init_identity(n, allocator);
            defer a.deinit(allocator);
            const b = try a.clone_into_complex(allocator);
            return b;
        }
        /// Initialise a complex matrix filled with a value
        pub fn init_fill_complex(n: usize, p: usize, value: T, allocator: Allocator) !Matrix(Complex(T)) {
            const a = try Matrix(T).init_fill(n, p, value, allocator);
            defer a.deinit(allocator);
            const b = try a.clone_into_complex(allocator);
            return b;
        }
        /// Print matrix
        pub fn print(self: Self) !void {
            var n: usize = self.n;
            var p: usize = self.p;
            if (n > 30) {
                n = 30;
            }
            if (p > 7) {
                p = 7;
            }
            for (0..n) |i| {
                if (i == 0) {
                    std.debug.print("⎡ ", .{});
                } else if (i < (n - 1)) {
                    std.debug.print("⎢ ", .{});
                } else {
                    std.debug.print("⎣ ", .{});
                }
                for (0..p) |j| {
                    if (((p < self.p) and (j == (p - 1))) or ((n < self.n) and (i == (n - 1)))) {
                        std.debug.print("... ", .{});
                    } else {
                        if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                            if (self.data[i][j].re > 0.0) {
                                std.debug.print(" ", .{});
                            }
                            std.debug.print("{d:.2}+{d:.2}i ", .{ self.data[i][j].re, self.data[i][j].im });
                        } else {
                            if (self.data[i][j] >= 0.0) {
                                std.debug.print(" ", .{});
                            }
                            std.debug.print("{d:.2} ", .{self.data[i][j]});
                        }
                    }
                    if ((i == 0) and (j == (p - 1))) {
                        std.debug.print("⎤", .{});
                    }
                }
                if ((i != 0) and (i < (n - 1))) {
                    std.debug.print("⎥", .{});
                }
                if (i == (n - 1)) {
                    std.debug.print("⎦", .{});
                }
                std.debug.print("\n", .{});
            }
        }
        /// Slice matrix
        pub fn slice(self: Self, row_indexes: []const usize, column_indexes: []const usize, allocator: Allocator) !Self {
            var S: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                S = try Matrix(F).init_complex(row_indexes.len, column_indexes.len, allocator);
            } else {
                S = try Matrix(T).init(row_indexes.len, column_indexes.len, allocator);
            }
            var idx_i: usize = 0;
            for (row_indexes) |i| {
                var idx_j: usize = 0;
                for (column_indexes) |j| {
                    S.data[idx_i][idx_j] = self.data[i][j];
                    idx_j += 1;
                }
                idx_i += 1;
            }
            // std.debug.print("SLICE:\n", .{});
            // try S.print();
            return S;
        }
        /// Matrix multiplication: A*B
        pub fn mult(self: Self, b: Self, allocator: Allocator) !Self {
            if (self.p != b.n) {
                return MatrixError.IncompatibleMatrices;
            }
            const n: usize = self.n;
            const p: usize = b.p;
            var product: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                product = try Matrix(F).init_complex(n, p, allocator);
            } else {
                product = try Matrix(T).init(n, p, allocator);
            }
            for (0..n) |i| {
                for (0..p) |j| {
                    var dot_product: T = multiply(T, self.data[i][0], b.data[0][j]);
                    for (1..self.p) |k| {
                        dot_product = add(T, dot_product, multiply(T, self.data[i][k], b.data[k][j]));
                    }
                    product.data[i][j] = dot_product;
                }
            }
            return product;
        }
        /// Matrix multiplication with transpose of the second matrix: A*(B^T)
        pub fn mult_bt(self: Self, b: Self, allocator: Allocator) !Self {
            if (self.p != b.p) {
                // transposed b
                return MatrixError.IncompatibleMatrices;
            }
            const n: usize = self.n;
            const p: usize = b.n; // transposed b
            var product: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                product = try Matrix(F).init_complex(n, p, allocator);
                for (0..n) |i| {
                    for (0..p) |j| {
                        var dot_product: T = multiply(T, self.data[i][0], b.data[j][0].conjugate()); // transposed b (conjugate transpose)
                        for (1..self.p) |k| {
                            dot_product = add(T, dot_product, multiply(T, self.data[i][k], b.data[j][k].conjugate())); // transposed b (conjugate transpose)
                        }
                        product.data[i][j] = dot_product;
                    }
                }
            } else {
                product = try Matrix(T).init(n, p, allocator);
                for (0..n) |i| {
                    for (0..p) |j| {
                        var dot_product: T = multiply(T, self.data[i][0], b.data[j][0]); // transposed b
                        for (1..self.p) |k| {
                            dot_product = add(T, dot_product, multiply(T, self.data[i][k], b.data[j][k])); // transposed b
                        }
                        product.data[i][j] = dot_product;
                    }
                }
            }
            return product;
        }
        /// Matrix multiplication with transpose of the first matrix: (A^T)*B
        pub fn mult_at(self: Self, b: Self, allocator: Allocator) !Self {
            if (self.n != b.n) {
                // transposed a
                return MatrixError.IncompatibleMatrices;
            }
            const n: usize = self.p; // transposed a
            const p: usize = b.p;
            var product: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                product = try Matrix(F).init_complex(n, p, allocator);
                for (0..n) |i| {
                    for (0..p) |j| {
                        var dot_product: T = multiply(T, self.data[0][i].conjugate(), b.data[0][j]); // transposed a (conjugate transpose)
                        for (1..self.n) |k| {
                            dot_product = add(T, dot_product, multiply(T, self.data[k][i].conjugate(), b.data[k][j])); // transposed a (conjugate transpose)
                        }
                        product.data[i][j] = dot_product;
                    }
                }
            } else {
                product = try Matrix(T).init(n, p, allocator);
                for (0..n) |i| {
                    for (0..p) |j| {
                        var dot_product: T = multiply(T, self.data[0][i], b.data[0][j]); // transposed a
                        for (1..self.n) |k| {
                            dot_product = add(T, dot_product, multiply(T, self.data[k][i], b.data[k][j])); // transposed a
                        }
                        product.data[i][j] = dot_product;
                    }
                }
            }
            return product;
        }
        /// Pivot: rearrange the rows so that the largest value in the ith column is in ith row.
        /// Yields a slice with number of elements equal to the numbrt of rows.
        /// In square matrices this means that the largest values per row are in the diagonals.
        pub fn pivot(self: Self, allocator: Allocator) ![]const usize {
            // Instatiate output slice
            var row_indexes = try allocator.alloc(usize, self.n);
            for (0..self.n) |i| {
                row_indexes[i] = i;
            }
            for (row_indexes) |i| {
                var i_max: usize = i;
                for (row_indexes[i..]) |j| {
                    if (less_than(T, absolute(T, self.data[i_max][i]), absolute(T, self.data[j][i]))) {
                        i_max = j;
                    }
                }
                row_indexes[i] = i_max;
                row_indexes[i_max] = i;
            }
            return row_indexes;
        }
        /// Forbenius norm (sum(xij^2))
        // TODO: Add more element-wise norms at some point, e.g. Max norm
        pub fn norm_forbenius(self: Self) !T {
            var sum = subtract(T, self.data[0][0], self.data[0][0]);
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    sum = add(T, sum, multiply(T, absolute(T, self.data[i][j]), absolute(T, self.data[i][j])));
                }
            }
            return square_root(T, sum);
        }
        /// Householder reflection
        /// Applicable to column vectors
        pub fn reflect_householder(self: Self, allocator: Allocator) !Self {
            // Make sure self is a column vector
            if (self.p != 1) {
                return MatrixError.NotAColumnVector;
            }
            const n = self.n;
            // Define the Forbenius norm
            const frobenius_norm: T = try self.norm_forbenius();
            // Define the normal vector from which the input vector, self gets projected into
            var normal_vector = try self.clone(allocator);
            defer normal_vector.deinit(allocator);
            var divisor = self.data[0][0];
            // std.debug.print("self:\n", .{});
            // try self.print();
            // std.debug.print("frobenius_norm={any}\n", .{frobenius_norm});
            // std.debug.print("divisor={any}\n", .{divisor});
            // std.debug.print("self.data[0][0]={any}\n", .{self.data[0][0]});
            if (greater_than(T, self.data[0][0], as(f64, T, 0.0)) or equal_to(T, self.data[0][0], as(f64, T, 0.0))) {
                divisor = add(T, divisor, frobenius_norm);
            } else {
                divisor = subtract(T, divisor, frobenius_norm);
            }
            // std.debug.print("Final divisor={any}\n", .{divisor});
            // std.debug.print("Before dividing by divisor | normal_vector:\n", .{});
            // try normal_vector.print();
            for (0..n) |i| {
                normal_vector.data[i][0] = divide(T, self.data[i][0], divisor);
            }
            normal_vector.data[0][0] = as(f64, T, 1.0); // set the origin, i.e. first element as 1
            // std.debug.print("Final normal_vector:\n", .{});
            // try normal_vector.print();
            // Instantiate the output matrix, H
            var H: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                H = try Matrix(F).init_identity_complex(n, allocator);
            } else {
                H = try Matrix(T).init_identity(n, allocator);
            }
            // Define the multiplier
            const M = try normal_vector.mult_at(normal_vector, allocator);
            // std.debug.print("M:\n", .{});
            // try M.print();
            defer M.deinit(allocator);
            const multiplier = divide(T, as(f64, T, 2.00), M.data[0][0]); // 2 / vT*v
            // std.debug.print("M:\n", .{});
            // try M.print();
            // Define the v*vT matrix
            var S = try normal_vector.mult_bt(normal_vector, allocator);
            defer S.deinit(allocator);
            // std.debug.print("S:\n", .{});
            // try S.print();
            // Build the Householder matrix
            for (0..n) |i| {
                for (0..n) |j| {
                    H.data[i][j] = subtract(T, H.data[i][j], multiply(T, multiplier, S.data[i][j]));
                }
            }
            // std.debug.print("H:\n", .{});
            // try H.print();
            return H;
        }
        /// Gaussian elimination (Ref: my interpretation of using the elementerary row opertaions to convert self into reduced row and column echelon form)
        /// Additionally updates the determinant of self
        /// Applicable for square and non-square matrices
        /// Note that non-square matrices do not have determinants
        pub fn gaussian_elimination(self: *Self, b: Self, allocator: Allocator) ![2]Self {
            // Make sure the two matrices have the same number of rows
            if (self.n != b.n) {
                return MatrixError.IncompatibleMatrices;
            }
            var determinant: T = as(f64, T, 1.00);
            // Define the pivot
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            // std.debug.print("row_indexes={any}\n", .{row_indexes});
            // Instatiate the pivoted reduced row echelon form matrices of self and b
            var self_echelon: Self = undefined;
            var b_echelon: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                self_echelon = try Matrix(F).init_complex(self.n, self.p, allocator);
                b_echelon = try Matrix(F).init_complex(b.n, b.p, allocator);
            } else {
                self_echelon = try Matrix(T).init(self.n, self.p, allocator);
                b_echelon = try Matrix(T).init(b.n, b.p, allocator);
            }
            for (row_indexes, 0..) |i_pivot, i| {
                if (i_pivot != i) {
                    determinant = negative(T, determinant);
                }
                for (0..self.p) |j| {
                    self_echelon.data[i][j] = self.data[i_pivot][j];
                }
                for (0..b.p) |j| {
                    b_echelon.data[i][j] = b.data[i_pivot][j];
                }
            }
            // Perform elementary row operations to convert self into an upper-triangular matrix, with diagonal of ones
            // std.debug.print("[BEFORE]\n", .{});
            // try self_echelon.print();
            // Forward: from the upper-left corner to the lower-right corner
            for (0..self_echelon.n) |i| {
                const a_ii = self_echelon.data[i][i];
                if (less_than(T, absolute(T, a_ii), as(f64, T, 0.000001))) {
                    return MatrixError.SingularMatrix;
                }
                determinant = multiply(T, determinant, a_ii);
                // Set the digonals as one
                for (0..self_echelon.p) |j| {
                    if (!equal_to(T, self_echelon.data[i][j], as(f64, T, 0.0))) {
                        self_echelon.data[i][j] = divide(T, self_echelon.data[i][j], a_ii);
                    }
                }
                for (0..b_echelon.p) |j| {
                    if (!equal_to(T, b_echelon.data[i][j], as(f64, T, 0.0))) {
                        b_echelon.data[i][j] = divide(T, b_echelon.data[i][j], a_ii);
                    }
                }
                if ((i + 1) == self_echelon.n) {
                    break;
                }
                // Subtract the product of the current diagonal value which is one and the value of the value below it to get a zero,
                // and do this for the whole row below it, where the values below the values to the left of the current diagonal value
                // are zero from previous iterations which also render all values below them to zero.
                for (0..(i + 1)) |k| {
                    const a_i_1k: T = self_echelon.data[i + 1][k];
                    for (0..self_echelon.p) |j| {
                        self_echelon.data[i + 1][j] = subtract(T, self_echelon.data[i + 1][j], multiply(T, a_i_1k, self_echelon.data[k][j]));
                    }
                    for (0..b_echelon.p) |j| {
                        b_echelon.data[i + 1][j] = subtract(T, b_echelon.data[i + 1][j], multiply(T, a_i_1k, b_echelon.data[k][j]));
                    }
                }
                // std.debug.print("Iteration {any}\n", .{i});
                // try self_echelon.print();
            }
            // std.debug.print("[AFTER]\n", .{});
            // try self_echelon.print();
            // Update the determinant field of self if self is square
            if (self.n == self.p) {
                // std.debug.print("determinant={any}\n", .{determinant});
                self.determinant = determinant;
            }
            // Reverse: from the lower-right corner to the upper-left corner
            for (0..self_echelon.n) |i_inverse| {
                const i: usize = self_echelon.n - (i_inverse + 1);
                if (i == 0) {
                    break;
                }
                for (0..(i_inverse + 1)) |k_inverse| {
                    const k = self_echelon.n - (k_inverse + 1);
                    const a_i_1k = self_echelon.data[i - 1][k];
                    for (0..self_echelon.p) |j| {
                        self_echelon.data[i - 1][j] = subtract(T, self_echelon.data[i - 1][j], multiply(T, a_i_1k, self_echelon.data[k][j]));
                    }
                    for (0..b_echelon.p) |j| {
                        b_echelon.data[i - 1][j] = subtract(T, b_echelon.data[i - 1][j], multiply(T, a_i_1k, b_echelon.data[k][j]));
                    }
                }
                // std.debug.print("Iteration {any}\n", .{i});
                // try self_echelon.print();
            }
            // std.debug.print("[FINAL self_echelon]\n", .{});
            // try self_echelon.print();
            // std.debug.print("[FINAL b_echelon]\n", .{});
            // try b_echelon.print();
            if (self_echelon.n == self_echelon.p) {
                // Update the determinants of the 2 ouput matrices just for completeness if they are square matrices
                self_echelon.determinant = as(f64, T, 1.00);
                for (0..self_echelon.n) |i| {
                    if (i < self_echelon.p) {
                        self_echelon.determinant.? = multiply(T, self_echelon.determinant.?, self_echelon.data[i][i]);
                    }
                }
            }
            if (b_echelon.n == b_echelon.p) {
                b_echelon.determinant = as(f64, T, 1.00);
                for (0..b_echelon.n) |i| {
                    if (i < b_echelon.p) {
                        b_echelon.determinant.? = multiply(T, b_echelon.determinant.?, b_echelon.data[i][i]);
                    }
                }
            }
            return .{ self_echelon, b_echelon };
        }
        /// LU decomposition (Ref: https://rosettacode.org/wiki/LU_decomposition)
        /// Additionally updates the determinant of self
        /// Applicable to square matrices only
        pub fn lu(self: *Self, allocator: Allocator) ![3]Self {
            // Make sure the matrix is square
            if (self.n != self.p) {
                return MatrixError.NonSquareMatrix;
            }
            // Instantiate the permutation, L, and U matrices
            var P: Self = undefined;
            var L: Self = undefined;
            var U: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                P = try Matrix(F).init_complex(self.n, self.p, allocator);
                L = try Matrix(F).init_complex(self.n, self.p, allocator);
                U = try Matrix(F).init_complex(self.n, self.p, allocator);
            } else {
                P = try Matrix(T).init(self.n, self.p, allocator);
                L = try Matrix(T).init(self.n, self.p, allocator);
                U = try Matrix(T).init(self.n, self.p, allocator);
            }
            // Populate with zeros
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    P.data[i][j] = as(f64, T, 0.0);
                    L.data[i][j] = as(f64, T, 0.0);
                    U.data[i][j] = as(f64, T, 0.0);
                }
            }
            // Define the permutation matrix
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            for (0..self.n) |i| {
                P.data[row_indexes[i]][i] = as(f64, T, 1.0);
            }
            // Decompose
            for (0..self.p) |j| {
                L.data[j][j] = as(f64, T, 1.0);
                for (row_indexes[0..(j + 1)], 0..) |i_a, i| {
                    var s1: T = as(f64, T, 0.0);
                    for (0..i) |k| {
                        s1 = add(T, s1, multiply(T, U.data[k][j], L.data[i][k]));
                    }
                    U.data[i][j] = subtract(T, self.data[i_a][j], s1);
                }
                for (row_indexes[j..self.n], j..self.n) |i_a, i| {
                    var s2 = as(f64, T, 0.0);
                    for (0..j) |k| {
                        s2 = add(T, s2, multiply(T, U.data[k][j], L.data[i][k]));
                    }
                    if (absolute(T, U.data[j][j]) < as(f64, T, 0.000001)) {
                        return MatrixError.SingularMatrix;
                    }
                    L.data[i][j] = divide(T, subtract(T, self.data[i_a][j], s2), U.data[j][j]);
                }
            }
            // Append the determinants of each matrix including self, P, L, and U
            P.determinant = as(f64, T, 1.0);
            L.determinant = as(f64, T, 1.0);
            self.determinant = as(f64, T, 1.0);
            for (0..self.n) |i| {
                self.determinant.? = multiply(T, self.determinant.?, U.data[i][i]);
            }
            return .{ P, L, U };
        }
        /// QR decomposition (Ref: https://rosettacode.org/wiki/QR_decomposition)
        /// Applicable for square and non-square matrices
        /// Using Householder reflection (looking into using Givens rotation)
        /// Note that the determinant of self is not updated here
        pub fn qr(self: Self, allocator: Allocator) ![2]Self {
            // Make sure that self has dimensions n >= p
            if (self.n < self.p) {
                return MatrixError.RowsLessThanColumns;
            }
            // Define the major axis, i.e. the row or column whichever is larger
            const n: usize = self.n;
            var p: usize = self.p;
            // If self is square then we do not iterate up to the final column (why?)
            if (self.n == self.p) {
                p -= 1;
            }
            // Initialise Q and R as identity matrices
            var Q: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                Q = try Matrix(F).init_identity_complex(n, allocator);
            } else {
                Q = try Matrix(T).init_identity(n, allocator);
            }
            var R = try self.clone(allocator);
            // Initialise the array of indexes for slicing and Housefolder reflection
            const indexes: []usize = try allocator.alloc(usize, n);
            defer allocator.free(indexes);
            for (0..n) |i| {
                indexes[i] = i;
            }
            // Iterate per column
            var H: Self = undefined;
            defer H.deinit(allocator);
            for (0..p) |j| {
                // Define the Householder reflection matrix for the current iteration
                if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                    const F: type = @TypeOf(self.data[0][0].re);
                    H = try Matrix(F).init_identity_complex(n, allocator);
                } else {
                    H = try Matrix(T).init_identity(n, allocator);
                }
                const a = try R.slice(indexes[j..], indexes[j..(j + 1)], allocator);
                const h = try a.reflect_householder(allocator);
                defer a.deinit(allocator);
                defer h.deinit(allocator);
                // std.debug.print("a:\n", .{});
                // try a.print();
                // std.debug.print("h:\n", .{});
                // try h.print();
                // NOTE: Skip the Householder transformation if the slice (a) is sparse which leads to NaNs
                if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                    if (std.math.isNan(h.data[0][0].re)) {
                        continue;
                    }
                } else {
                    if (std.math.isNan(h.data[0][0])) {
                        continue;
                    }
                }
                for (j..n, 0..) |H_i, h_i| {
                    for (j..n, 0..) |H_j, h_j| {
                        H.data[H_i][H_j] = h.data[h_i][h_j];
                    }
                }
                // Multiple Q by the H
                const Q_mult = try Q.mult(H, allocator);
                defer Q_mult.deinit(allocator);
                // Multiple H by R
                const R_mult = try H.mult(R, allocator);
                defer R_mult.deinit(allocator);
                // Update Q and R
                for (0..n) |Q_i| {
                    for (0..n) |Q_j| {
                        Q.data[Q_i][Q_j] = Q_mult.data[Q_i][Q_j];
                        if (Q_j < R.p) {
                            R.data[Q_i][Q_j] = R_mult.data[Q_i][Q_j];
                        }
                    }
                }
            }
            return .{ Q, R };
        }
        /// Cholesky decomposition (Ref: https://rosettacode.org/wiki/Cholesky_decomposition)
        /// Applicable to square symmetric matrices
        /// This is a **very fast** algorithm but requires Hermitian positive-definite matrix!
        pub fn chol(self: *Self, allocator: Allocator) !Self {
            // Make sure the self is square
            if (self.n != self.p) {
                return MatrixError.NonSquareMatrix;
            }
            // Make sure the matrix is Hermitian
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    if (self.data[i][j] != self.data[j][i]) {
                        return MatrixError.NonHermitian;
                    }
                }
            }
            const n = self.p;
            var L: Self = undefined;
            if ((T == Complex(f16)) or (T == Complex(f32)) or (T == Complex(f64)) or (T == Complex(f128))) {
                const F: type = @TypeOf(self.data[0][0].re);
                L = try Matrix(F).init_fill_complex(n, n, as(f64, T, 0.0), allocator);
            } else {
                L = try Matrix(T).init_fill(n, n, as(f64, T, 0.0), allocator);
            }
            var determinant = as(f64, T, 1.0);
            for (0..n) |i| {
                for (0..(i + 1)) |j| {
                    var sum = as(f64, T, 0.0);
                    for (0..j) |k| {
                        sum = add(T, sum, multiply(T, L.data[i][k], L.data[j][k]));
                    }
                    if (i == j) {
                        const L_ij_squared = add(T, self.data[i][i], sum);
                        if (L_ij_squared < as(f64, T, 0.0)) {
                            return MatrixError.NotPositiveSemiDefinite;
                        }
                        L.data[i][j] = square_root(T, L_ij_squared);
                    } else {
                        if (L.data[j][j] == as(f64, T, 0.0)) {
                            return MatrixError.NotPositiveSemiDefinite;
                        }
                        L.data[i][j] = multiply(T, divide(T, as(f64, T, 1.0), L.data[j][j]), add(T, self.data[i][j], sum));
                    }
                    // std.debug.print("@@@@@@@@@@@@@@@@@@@@@\n", .{});
                    // std.debug.print("i={any}; j={any}:\n", .{ i, j });
                    // std.debug.print("A:\n", .{});
                    // try self.print();
                    // std.debug.print("L:\n", .{});
                    // try L.print();
                }
                determinant = multiply(T, determinant, multiply(T, L.data[i][i], L.data[i][i])); // deteterminant is equal to the product of the squares of the diagonal of L
            }
            // Update the determinant in self and L
            self.determinant = determinant;
            L.determinant = determinant;
            return L;
        }
        /// Eigen decompostion via QR algorithm
        /// Applicable to square matrices
        /// This is a slow implementation of the QR algorithm which uses non-parallelisable Householder reflection.
        /// Where are the complex components? They come from solving the quadratic equation in finiding Wilkinson's shift.
        pub fn eigendecomposition_QR(self: Matrix(T), comptime F: type, allocator: Allocator) ![2]Matrix(Complex(F)) {
            // If self is a complex matrix then F is the type of the floats inside the complex numbers
            // otherwise T == F (if the matrix is not complex)
            if (self.n != self.p) {
                return MatrixError.NonSquareMatrix;
            }
            const C: type = Complex(F);
            var A: Matrix(C) = undefined;
            if (T == C) {
                A = try self.clone(allocator);
            } else {
                A = try self.clone_into_complex(allocator);
            }
            defer A.deinit(allocator);
            const max_iter: usize = 10_000;
            var eigenvalues = try Matrix(C).init(self.n, 1, allocator);
            var eigenvectors = try Matrix(F).init_complex(self.n, self.p, allocator);
            var shifter: C = Complex(F).init(0.0, 0.0);
            var n_eigens_finished: usize = 0;
            for (0..max_iter) |_| {
                // Define the shifter (Wilkinson's shift)
                if (A.n >= 2) {
                    const A_a: C = A.data[A.n - 2][A.n - 2];
                    const A_b: C = A.data[A.n - 2][A.n - 1];
                    const A_c: C = A.data[A.n - 1][A.n - 2];
                    const A_d: C = A.data[A.n - 1][A.n - 1];
                    // std.debug.print("A_a={any}\n", .{A_a});
                    // std.debug.print("A_b={any}\n", .{A_b});
                    // std.debug.print("A_c={any}\n", .{A_c});
                    // std.debug.print("A_d={any}\n", .{A_d});
                    if (std.math.isNan(A_d.re)) {
                        return MatrixError.NullDataInMatrix;
                    }

                    const a: C = as(f64, C, 1.00);
                    const b: C = add(C, A_a, A_d);
                    const c: C = subtract(C, multiply(C, A_a, A_d), multiply(C, A_b, A_c));
                    const b2_4ac: C = subtract(C, multiply(C, b, b), multiply(C, multiply(C, as(f64, C, 4.00), a), c));
                    const x_1: C = divide(C, add(C, negative(C, b), square_root(C, b2_4ac)), multiply(C, as(f64, C, 2.00), a));
                    const x_2: C = divide(C, subtract(C, negative(C, b), square_root(C, b2_4ac)), multiply(C, as(f64, C, 2.00), a));
                    // Which root is closest to A_d?
                    const diff_1 = absolute(C, subtract(C, A_d, x_1));
                    const diff_2 = absolute(C, subtract(C, A_d, x_2));

                    if (less_than(C, diff_1, diff_2)) {
                        shifter = x_1;
                    } else {
                        shifter = x_2;
                    }
                }
                // std.debug.print("shifter={any}\n", .{shifter});
                // Subtract the shifter from A prior to QR decomposition
                for (0..A.n) |i| {
                    A.data[i][i] = subtract(C, A.data[i][i], shifter);
                }
                // QR decomposition
                const QR = try A.qr(allocator);
                defer QR[0].deinit(allocator);
                defer QR[1].deinit(allocator);
                // std.debug.print("QR[0]:\n", .{});
                // try QR[0].print();
                // std.debug.print("QR[1]:\n", .{});
                // try QR[1].print();
                // Reconsititute A
                A = try QR[1].mult(QR[0], allocator);
                // std.debug.print("A:\n", .{});
                // try A.print();
                // Add back the shifter into A
                for (0..A.n) |i| {
                    A.data[i][i] = add(C, A.data[i][i], shifter);
                }
                // Go to the next eigenvalue if all row below n_eigens_finished(th) column are all zero
                var n_threshold_passed: usize = 0;
                for (1..A.n) |i| {
                    var max_Aii_Ajj = absolute(C, A.data[0][0]);
                    if (greater_than(C, absolute(C, A.data[i][i]), max_Aii_Ajj)) {
                        max_Aii_Ajj = absolute(C, A.data[i][i]);
                    }
                    if (equal_to(C, max_Aii_Ajj, as(f64, C, 0.0))) {
                        continue;
                    }
                    const Aij_over_max_Aii_Ajj = divide(C, absolute(C, A.data[i][0]), max_Aii_Ajj);
                    // std.debug.print("Aij_over_max_Aii_Ajj={any}\n", .{Aij_over_max_Aii_Ajj});
                    if (less_than(C, Aij_over_max_Aii_Ajj, as(f64, C, 0.0001))) {
                        n_threshold_passed += 1;
                        // A.data[i][0] = as(f64, C, 0.0);
                    }
                }
                std.debug.print("A.n={any}; n_threshold_passed={any}\n", .{ A.n, n_threshold_passed });
                if (n_threshold_passed == (A.n - 1)) {
                    eigenvalues.data[n_eigens_finished][0] = A.data[0][0];
                    n_eigens_finished += 1;
                    var row_indexes: []usize = try allocator.alloc(usize, A.n - 1);
                    var col_indexes: []usize = try allocator.alloc(usize, A.n - 1);
                    for (1..A.n) |i| {
                        row_indexes[i - 1] = i;
                        col_indexes[i - 1] = i;
                    }
                    // std.debug.print("row_indexes={any}\n", .{row_indexes});
                    A = try A.slice(row_indexes, col_indexes, allocator);
                    // std.debug.print("A:\n", .{});
                    // try A.print();
                }

                if (A.n < 2) {
                    break;
                }
            }
            // Update with the final eigenvalue
            eigenvalues.data[n_eigens_finished][0] = A.data[A.n - 1][A.n - 1];
            // std.debug.print("n_eigens_finished={any}\n", .{n_eigens_finished});
            // std.debug.print("A.data[A.n - 1][A.n - 1]={any}:\n", .{A.data[A.n - 1][A.n - 1]});
            // std.debug.print("eigenvalues:\n", .{});
            // try eigenvalues.print();

            // Extract eigenvectors
            // We will use Gaussian elimination to solve for the eigenvector, v, where (A-lI)v = b, where l is an eigenvalue, I is an identity matrix, and b is a vector of ones
            // (Note: mathematically b should be zeros, but algorithmically to get the eigenvector we set b to onees instead of zeros)
            // It is critical to normalise (divide by sqrt(sum(v^2)) the eigenvectors so that they map to the basis vectors, i.e. squares sum up to 1).
            var A_minus_lambda: Matrix(C) = undefined;
            defer A_minus_lambda.deinit(allocator);
            if (T == C) {
                A = try self.clone(allocator);
                A_minus_lambda = try self.clone(allocator);
            } else {
                A = try self.clone_into_complex(allocator);
                A_minus_lambda = try self.clone_into_complex(allocator);
            }
            const identity = try Matrix(F).init_fill_complex(self.n, 1, 1.00, allocator);
            defer identity.deinit(allocator);
            for (0..eigenvalues.n) |j| {
                const lambda: C = eigenvalues.data[j][0];
                for (0..A_minus_lambda.n) |ix| {
                    A_minus_lambda.data[ix][ix] = subtract(C, A.data[ix][ix], lambda);
                }
                const V = try A_minus_lambda.gaussian_elimination(identity, allocator);
                defer V[0].deinit(allocator);
                defer V[1].deinit(allocator);
                var norm: C = multiply(C, V[1].data[0][0], V[1].data[0][0]);
                for (1..V[1].n) |i| {
                    norm = add(C, norm, multiply(C, V[1].data[i][0], V[1].data[i][0]));
                }
                norm = square_root(C, norm);
                for (0..eigenvectors.n) |i| {
                    // eigenvectors.data[i][j] = V[1].data[i][0];
                    eigenvectors.data[i][j] = divide(C, V[1].data[i][0], norm);
                }
                // const TEST = try A_minus_lambda.mult(V[1], allocator);
                // defer TEST.deinit(allocator);
                // std.debug.print("i:{any}\n", .{j});
                // try A_minus_lambda.print();
                // try V[0].print();
                // try V[1].print();
                // std.debug.print("i:{any}\n", .{j});
                // std.debug.print("norm={any}\n", .{norm});
                // std.debug.print("A:\n", .{});
                // try A.print();
                // std.debug.print("lambda={any}\n", .{lambda});
                // std.debug.print("A_minus_lambda:\n", .{});
                // try A_minus_lambda.print();
                // std.debug.print("v:\n", .{});
                // try V[1].print();
                // std.debug.print("eigenvectors:\n", .{});
                // try eigenvectors.print();
                // try TEST.print();
            }
            // std.debug.print("Eigenvalues:\n", .{});
            // try eigenvalues.print();
            // std.debug.print("Eigenvectors:\n", .{});
            // try eigenvectors.print();
            return .{ eigenvalues, eigenvectors };
        }
        /// Singular valude decomposition (Ref: https://builtin.com/articles/svd-algorithm)
        /// Generalisation of eigendecomposition on non-square matrices
        /// U = eigenvectors of MM*; V* = eigenvectors of M*M; and diagonals of S = square roots of the eigenvalues of M*M and MM*
        pub fn svd(self: Matrix(T), comptime F: type, allocator: Allocator) ![3]Matrix(Complex(F)) {
            const C: type = Complex(F);
            var M_: Matrix(C) = try self.clone_into_complex(allocator);
            const M: Matrix(C) = try self.clone_into_complex(allocator);
            defer M.deinit(allocator);
            const MMstar = try M_.mult_bt(M, allocator);
            std.debug.print("MMstar:\n", .{});
            try MMstar.print();

            const MstarM = try M_.mult_at(M, allocator);
            std.debug.print("MstarM:\n", .{});
            try MstarM.print();

            const left_eigens = try MMstar.eigendecomposition_QR(F, allocator);
            defer left_eigens[0].deinit(allocator);
            defer left_eigens[1].deinit(allocator);
            std.debug.print("Eigen values for SVD:\n", .{});
            try left_eigens[0].print();

            return .{ MMstar, MstarM, left_eigens[0] };
        }
    };
}

test "Initialisations, cloning and slicing" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Initialisations & cloning", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});

    // Initialisation
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 4;
    const p: usize = 4;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    try expect(a.n == n);
    try expect(a.p == p);
    std.debug.print("a.data[0][0]={?}\n", .{a.data[0][0]});
    std.debug.print("a={any}\n", .{a});

    // Float type assertion
    const type_error = Matrix(u8).init(10, 123, allocator);
    try expect(type_error == MatrixError.TIsNotAFloatType);

    // Populate with data from Example2 in https://rosettacode.org/wiki/LU_decomposition
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    const contents = [16]f64{ 11.0, 9.0, 24.0, 2.0, 1.0, 5.0, 2.0, 6.0, 3.0, 17.0, 18.0, 1.0, 2.0, 5.0, 7.0, 1.0 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    std.debug.print("a:\n", .{});
    try a.print();
    std.debug.print("b:\n", .{});
    try b.print();

    // Identity matrix
    const identity = try Matrix(f64).init_identity(n, allocator);
    defer identity.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                try expect(identity.data[i][j] == 1.00);
            } else {
                try expect(identity.data[i][j] == 0.00);
            }
        }
    }

    // Pre-filled matrix with a constant value
    const value: f64 = 3.1416;
    const matrix_with_constant = try Matrix(f64).init_fill(n, p, value, allocator);
    defer matrix_with_constant.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try expect(matrix_with_constant.data[i][j] == value);
        }
    }

    // Clone
    const a_copy = try a.clone(allocator);
    defer a_copy.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try expect(a_copy.data[i][j] == a.data[i][j]);
        }
    }

    // Slice
    const row_indexes = [2]usize{ 2, 0 };
    const column_indexes = [3]usize{ 1, 0, 2 };
    const S = try a.slice(&row_indexes, &column_indexes, allocator);
    std.debug.print("S:\n", .{});
    try S.print();
    var idx_i: usize = 0;
    for (row_indexes) |i| {
        var idx_j: usize = 0;
        for (column_indexes) |j| {
            try expect(a.data[i][j] == S.data[idx_i][idx_j]);
            idx_j += 1;
        }
        idx_i += 1;
    }
}

test "Complex matrices" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Complex matrices", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});

    // Initialisation
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 4;
    const p: usize = 4;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    try expect(a.n == n);
    try expect(a.p == p);
    std.debug.print("a.data[0][0]={?}\n", .{a.data[0][0]});
    std.debug.print("a={any}\n", .{a});

    // Clone real matrix into a complex matrix with 0.0 imaginary component
    const a_complex = try a.clone_into_complex(allocator);
    defer a_complex.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try expect(a_complex.data[i][j].re == a.data[i][j]);
            try expect(a_complex.data[i][j].im == 0.0);
        }
    }

    // Initialise a complex matrix
    const b_complex = try Matrix(f64).init_complex(n, p, allocator);
    defer b_complex.deinit(allocator);
    const C: type = @TypeOf(b_complex.data[0][0]);
    try expect(C == Complex(f64));

    // Identity matrix
    const identity_complex = try Matrix(f64).init_identity_complex(n, allocator);
    defer identity_complex.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                try expect(identity_complex.data[i][j].re == 1.00);
                try expect(identity_complex.data[i][j].im == 0.00);
            } else {
                try expect(identity_complex.data[i][j].re == 0.00);
                try expect(identity_complex.data[i][j].im == 0.00);
            }
        }
    }

    // Pre-filled matrix with a constant value
    const value: f64 = 3.1416;
    const matrix_with_constant_complex = try Matrix(f64).init_fill_complex(n, p, value, allocator);
    defer matrix_with_constant_complex.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try expect(matrix_with_constant_complex.data[i][j].re == value);
            try expect(matrix_with_constant_complex.data[i][j].im == 0.0);
        }
    }
}

// TODO: test complex matrices
test "Matrix multiplication" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Matrix multiplication", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Matrix multiplication: A*B
    const more_contents = [20]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 };
    const a_3x5 = try Matrix(f32).init(3, 5, allocator);
    defer a_3x5.deinit(allocator);
    const a_5x3 = try Matrix(f32).init(5, 3, allocator);
    defer a_5x3.deinit(allocator);
    const b_5x3 = try Matrix(f32).init(5, 3, allocator);
    defer b_5x3.deinit(allocator);
    const b_3x2 = try Matrix(f32).init(3, 2, allocator);
    defer b_3x2.deinit(allocator);
    var counter: usize = 0;
    for (0..3) |i| {
        for (0..5) |j| {
            a_3x5.data[i][j] = more_contents[counter];
            counter += 1;
        }
    }
    counter = 0;
    for (0..5) |i| {
        for (0..3) |j| {
            a_5x3.data[i][j] = more_contents[counter];
            counter += 1;
        }
    }
    counter = 0;
    for (0..5) |i| {
        for (0..3) |j| {
            b_5x3.data[i][j] = more_contents[counter];
            counter += 1;
        }
    }
    counter = 0;
    for (0..3) |i| {
        for (0..2) |j| {
            b_3x2.data[i][j] = more_contents[counter];
            counter += 1;
        }
    }
    const c_axb = try a_3x5.mult(b_5x3, allocator);
    defer c_axb.deinit(allocator);
    const expected_a_3x5_mult_b_5x3 = [9]f32{ 135, 150, 165, 310, 350, 390, 485, 550, 615 };
    counter = 0;
    std.debug.print("c_axb={any}\n", .{c_axb});
    for (0..3) |i| {
        for (0..3) |j| {
            try expect(c_axb.data[i][j] == expected_a_3x5_mult_b_5x3[counter]);
            counter += 1;
        }
    }
    const c_axbT = try a_5x3.mult_bt(b_5x3, allocator);
    defer c_axbT.deinit(allocator);
    const expected_a_5x3_mult_b_5x3 = [25]f32{ 14, 32, 50, 68, 86, 32, 77, 122, 167, 212, 50, 122, 194, 266, 338, 68, 167, 266, 365, 464, 86, 212, 338, 464, 590 };
    counter = 0;
    std.debug.print("c_axbT={any}\n", .{c_axbT});
    for (0..5) |i| {
        for (0..5) |j| {
            try expect(c_axbT.data[i][j] == expected_a_5x3_mult_b_5x3[counter]);
            counter += 1;
        }
    }
    const c_aTxb = try a_3x5.mult_at(b_3x2, allocator);
    defer c_aTxb.deinit(allocator);
    const expected_a_3x5_mult_b_3x2 = [10]f32{ 74, 92, 83, 104, 92, 116, 101, 128, 110, 140 };
    counter = 0;
    std.debug.print("c_aTxb={any}\n", .{c_aTxb});
    for (0..5) |i| {
        for (0..2) |j| {
            try expect(c_aTxb.data[i][j] == expected_a_3x5_mult_b_3x2[counter]);
            counter += 1;
        }
    }
}

test "Pivot, norms, reflections, and rotations" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Pivot and norms", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 3;
    const p: usize = 3;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    // Populate with data from Example2 in https://rosettacode.org/wiki/LU_decomposition
    const contents = [9]f64{ 1, 3, 5, 2, 4, 7, 1, 1, 0 };
    // const contents = [9]f64{ 2, 9, 4, 7, 5, 3, 6, 1, 8 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
        }
    }
    std.debug.print("a\n", .{});
    try a.print();

    // Pivot
    const row_indexes = try a.pivot(allocator);
    defer allocator.free(row_indexes);
    std.debug.print("row_indexes={any}\n", .{row_indexes});
    const expected_row_indexes = [3]usize{ 1, 0, 2 };
    for (0..n) |i| {
        try expect(row_indexes[i] == expected_row_indexes[i]);
    }

    // Norms
    const norm_f = try a.norm_forbenius();
    std.debug.print("norm_f: {any}\n", .{norm_f});
    try expect(absolute(f64, norm_f - 10.29563) < 0.00001);

    //Reflection
    var b = try Matrix(f64).init(p, 1, allocator);
    defer b.deinit(allocator);
    const contents_b = [3]f64{ 12, 6, -4 };
    for (0..n) |i| {
        b.data[i][0] = contents_b[i];
    }

    const H = try b.reflect_householder(allocator);
    std.debug.print("H:\n", .{});
    try H.print();

    const expected_housholder_reflections = [9]f64{ -0.85714286, -0.42857143, 0.28571429, -0.42857143, 0.9010989, 0.06593407, 0.28571429, 0.06593407, 0.95604396 };
    for (0..n) |i| {
        for (0..n) |j| {
            const k = (i * n) + j;
            try expect(absolute(f64, H.data[i][j] - expected_housholder_reflections[k]) < 0.00001);
        }
    }
}

test "Gaussian elimination" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Gaussian elimination", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 5;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    const contents = [25]f64{ 12, 22, 35, 64, 2, 16, 72, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    const identity = try Matrix(f64).init_identity(n, allocator);
    defer identity.deinit(allocator);
    std.debug.print("a={any}\n", .{a});
    std.debug.print("b={any}\n", .{b});
    std.debug.print("identity={any}\n", .{identity});

    // Gaussian elimination
    var timer = try std.time.Timer.start();
    const echelons = try a.gaussian_elimination(identity, allocator);
    const time_elapsed = timer.read();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});

    std.debug.print("a={any}\n", .{a});

    defer echelons[0].deinit(allocator);
    defer echelons[1].deinit(allocator);
    const a_inverse = echelons[1];
    std.debug.print("echelons[0]\n", .{});
    try echelons[0].print();
    std.debug.print("echelons[0]={any}\n", .{echelons[0]});

    std.debug.print("a_inverse\n", .{});
    try a_inverse.print();
    std.debug.print("a_inverse={any}\n", .{a_inverse});

    const should_be_identity = try a.mult(a_inverse, allocator);
    defer should_be_identity.deinit(allocator);
    std.debug.print("should_be_identity\n", .{});
    try should_be_identity.print();
    for (0..n) |i| {
        for (0..n) |j| {
            var value = should_be_identity.data[i][j];
            if (i == j) {
                value -= 1.00;
            }
            if (value < 0.0) {
                value *= -1.0;
            }
            try expect(value < 0.00001);
        }
    }
}

test "LU decomposition" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("LU decomposition", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 5;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    const contents = [25]f64{ 12, 22, 35, 64, 2, 16, 72, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    const identity = try Matrix(f64).init_identity(n, allocator);
    defer identity.deinit(allocator);
    std.debug.print("a={any}\n", .{a});
    std.debug.print("b={any}\n", .{b});
    std.debug.print("identity={any}\n", .{identity});

    // LU decomposition
    var timer = try std.time.Timer.start();
    const out_test = try a.lu(allocator);
    const time_elapsed = timer.read();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    std.debug.print("a={any}\n", .{a});
    defer out_test[0].deinit(allocator);
    defer out_test[1].deinit(allocator);
    defer out_test[2].deinit(allocator);
    std.debug.print("P:\n", .{});
    try out_test[0].print();
    std.debug.print("L:\n", .{});
    try out_test[1].print();
    std.debug.print("U:\n", .{});
    try out_test[2].print();
    const PL = try out_test[0].mult(out_test[1], allocator);
    const PLU = try PL.mult(out_test[2], allocator);
    for (0..a.n) |i| {
        for (0..a.p) |j| {
            try expect(@abs(a.data[i][j] - PLU.data[i][j]) < 0.0000001);
        }
    }
    // Invert a, we let A be a
    // Given: Ax = b
    // LU decomposition: A = PLU
    // Now since: PA = LU, and Ax = b
    // Then: LUx = Pb
    // Now if B is an identity matrix then x becomes the inverse of A, i.e.: AX = I
    // Rewriting we have: LUX = PI -> LUX = P
    // We can now solve for X by gausian ellimination given the augmented matrix: LU|P, i.e. using elementary operations to convert LU into I we get P turned into X (the inverse of A)
    var LU = try out_test[1].mult(out_test[2], allocator);
    const out_gauss = try LU.gaussian_elimination(out_test[0], allocator);
    defer out_gauss[0].deinit(allocator);
    defer out_gauss[1].deinit(allocator);
    const should_be_identity = try a.mult(out_gauss[1], allocator);
    defer should_be_identity.deinit(allocator);
    std.debug.print("a\n", .{});
    try a.print();
    std.debug.print("a_inverse\n", .{});
    try out_gauss[1].print();
    std.debug.print("should_be_identity\n", .{});
    try should_be_identity.print();
    for (0..n) |i| {
        for (0..n) |j| {
            var value = should_be_identity.data[i][j];
            if (i == j) {
                value -= 1.00;
            }
            if (value < 0.0) {
                value *= -1.0;
            }
            try expect(value < 0.00001);
        }
    }
}

test "QR decomposition" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("QR decomposition", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 5;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    const contents = [25]f64{ 12, 22, 35, 64, 2, 16, 72, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    const identity = try Matrix(f64).init_identity(n, allocator);
    defer identity.deinit(allocator);
    std.debug.print("a={any}\n", .{a});
    std.debug.print("b={any}\n", .{b});
    std.debug.print("identity={any}\n", .{identity});

    // QR decomposition
    // Square matrix and using the inverse to test
    var timer = try std.time.Timer.start();
    const QR = try a.qr(allocator);
    var time_elapsed = timer.read();
    var Q = QR[0];
    defer Q.deinit(allocator);
    var R = QR[1];
    defer R.deinit(allocator);
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    std.debug.print("Q\n", .{});
    try Q.print();
    std.debug.print("R\n", .{});
    try R.print();

    const R_echelons = try R.gaussian_elimination(identity, allocator);
    const R_inv = R_echelons[1];
    defer R_inv.deinit(allocator);
    const a_inv = try R_inv.mult_bt(Q, allocator);
    defer a_inv.deinit(allocator);
    const should_be_another_identity = try a.mult(a_inv, allocator);
    defer should_be_another_identity.deinit(allocator);
    std.debug.print("should_be_another_identity\n", .{});
    try should_be_another_identity.print();
    for (0..n) |i| {
        for (0..n) |j| {
            var value = should_be_another_identity.data[i][j];
            if (i == j) {
                value -= 1.00;
            }
            if (value < 0.0) {
                value *= -1.0;
            }
            try expect(value < 0.00001);
        }
    }

    // Non-square matrix where there should be more rows than columns
    var A = try Matrix(f64).init(4, 3, allocator);
    const contents_A = [12]f64{ 12, -51, 4, 6, 167, -68, -4, 24, -41, 10, 2, -7 };
    for (0..4) |i| {
        for (0..3) |j| {
            const idx = (i * 3) + j;
            A.data[i][j] = contents_A[idx];
        }
    }
    std.debug.print("A:\n", .{});
    try A.print();
    timer.reset();
    const QR_A = try A.qr(allocator);
    time_elapsed = timer.read();
    var Q_A = QR_A[0];
    defer Q_A.deinit(allocator);
    var R_A = QR_A[1];
    defer R_A.deinit(allocator);
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    std.debug.print("Q_A\n", .{});
    try Q_A.print();
    std.debug.print("R_A\n", .{});
    try R_A.print();
    const expected_Q = [16]f64{ -0.6974858, 0.36350635, -0.30442756, -0.5373086, -0.3487429, -0.91624258, 0.04414177, -0.1921703, 0.2324953, -0.16109590, -0.95061037, 0.1278045, -0.5812382, 0.04909956, -0.04141614, 0.8111942 };
    const expected_R = [3 * 3]f64{ -17.20465, -18.25088, 15.46094, 0.00000, -175.31944, 70.01976, 0.00000, 0.00000, 35.04559 };
    for (0..4) |i| {
        for (0..4) |j| {
            try expect(absolute(f64, Q_A.data[i][j] - expected_Q[(i * 4) + j]) < 0.00001);
        }
    }
    for (0..3) |i| {
        for (0..3) |j| {
            try expect(absolute(f64, R_A.data[i][j] - expected_R[(i * 3) + j]) < 0.00001);
        }
    }
}

test "Cholesky decomposition" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Cholesky decomposition", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 4;
    var H = try Matrix(f64).init(n, n, allocator);
    defer H.deinit(allocator);
    const contents_H = [16]f64{ 18, 22, 54, 42, 22, 70, 86, 62, 54, 86, 174, 134, 42, 62, 134, 106 };
    for (0..4) |i| {
        for (0..4) |j| {
            H.data[i][j] = contents_H[(i * 4) + j];
        }
    }

    // Cholesky decomposition
    var timer = try std.time.Timer.start();
    const CHOL = try H.chol(allocator);
    const time_elapsed = timer.read();
    defer CHOL.deinit(allocator);
    std.debug.print("CHOL (det={any})\n", .{CHOL.determinant});
    try CHOL.print();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});

    const H_reconstructed = try CHOL.mult_bt(CHOL, allocator);
    defer H_reconstructed.deinit(allocator);

    std.debug.print("H\n", .{});
    try H.print();

    std.debug.print("H_reconstructed\n", .{});
    try H_reconstructed.print();

    for (0..4) |i| {
        for (0..4) |j| {
            try expect(@round(H.data[i][j] - H_reconstructed.data[i][j]) < 0.00001);
        }
    }
}

test "Eigen decomposition" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Eigen decomposition", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; // defer is autimatically executed after we return, i.e. when we go beyon the scope in which gpa was created
    const allocator = gpa.allocator();
    const n: usize = 5;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    const contents = [25]f64{ 912, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 943, 34, 5, 156, 12, 56, 997, 312, 546, 7, 28, 586, 970 };
    // const contents = [25]f64{ 12, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    const identity = try Matrix(f64).init_identity_complex(n, allocator);
    defer identity.deinit(allocator);
    std.debug.print("a:\n", .{});
    try a.print();
    std.debug.print("b={any}\n", .{b});
    std.debug.print("identity={any}\n", .{identity});

    // Eigenvalues and eigenvectors
    var timer = try std.time.Timer.start();
    const eigen_out = try a.eigendecomposition_QR(f64, allocator);
    const time_elapsed = timer.read();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    // std.debug.print("Eigenvalues: {any}\n", .{eigen_out[0].data});
    std.debug.print("eigenvalues:\n", .{});
    try eigen_out[0].print();
    std.debug.print("eigenvectors:\n", .{});
    try eigen_out[1].print();

    var P = try eigen_out[1].clone(allocator);
    std.debug.print("P:\n", .{});
    try P.print();
    var D = try Matrix(f64).init_identity_complex(5, allocator);
    for (0..5) |i| {
        D.data[i][i] = eigen_out[0].data[i][0];
    }
    std.debug.print("D:\n", .{});
    try D.print();

    // const P_inverse = try P.gaussian_elimination(identity, allocator);
    const QR = try P.qr(allocator);
    var Q = QR[0];
    defer Q.deinit(allocator);
    var R = QR[1];
    defer R.deinit(allocator);
    const R_echelons = try R.gaussian_elimination(identity, allocator);
    const R_inv = R_echelons[1];
    defer R_inv.deinit(allocator);
    const P_inverse = try R_inv.mult_bt(Q, allocator);
    defer P_inverse.deinit(allocator);
    std.debug.print("P_inverse:\n", .{});
    try P_inverse.print();

    // std.debug.print("P_inverse[1]:\n", .{});
    // try P_inverse[1].print();
    const PD = try P.mult(D, allocator);
    std.debug.print("PD:\n", .{});
    try PD.print();
    // const a_reconstituted = try PD.mult(P_inverse[1], allocator);
    const a_reconstituted = try PD.mult(P_inverse, allocator);
    std.debug.print("a_reconstituted:\n", .{});
    try a_reconstituted.print();
    const C: type = @TypeOf(a_reconstituted.data[0][0]);
    for (0..a.n) |i| {
        for (0..a.p) |j| {
            const abs_diff = absolute(C, subtract(C, as(f64, C, a.data[i][j]), a_reconstituted.data[i][j]));
            // std.debug.print("abs_diff={any}\n", .{abs_diff});
            try expect(less_than(C, abs_diff, as(f64, C, 1.0)));
        }
    }

    // Checking in R
    // x = c(912, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 943, 34, 5, 156, 12, 56, 997, 312, 546, 7, 28, 586, 970)
    // x = c(12, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970)
    // X = matrix(x, nrow=5, byrow=TRUE)
    // det(X)
    // E = eigen(X)
    // P = E$vectors
    // D = diag(E$values)
    // Y = P %*% D %*% solve(P)
    // sum(abs(X - Y)) < 0.000001
    // L = diag(E$values[1], nrow=5)
    // (X - L)
    // solve(X - L)
    // solve(X - L) %*% (X-L)

}

test "Singular value decomposition" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Singular value decomposition", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    // Preliminaries
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; // defer is autimatically executed after we return, i.e. when we go beyon the scope in which gpa was created
    const allocator = gpa.allocator();
    const n: usize = 10;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    const contents = [50]f64{ 912, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 943, 34, 5, 156, 12, 56, 997, 312, 546, 7, 28, 586, 970, 12, 22, 35, 64, 2, 16, 972, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
        }
    }
    const identity = try Matrix(f64).init_identity_complex(n, allocator);
    defer identity.deinit(allocator);
    std.debug.print("a:\n", .{});
    try a.print();
    std.debug.print("identity={any}\n", .{identity});

    // Eigenvalues and eigenvectors
    // var timer = try std.time.Timer.start();
    // const svd_out = try a.svd(f64, allocator);
    // const time_elapsed = timer.read();
    // std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    // try svd_out[0].print();
}
