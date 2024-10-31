const std: type = @import("std");
const expect = std.testing.expect;
const Allocator: type = std.mem.Allocator;

const MatrixError = error{
    IncompatibleMatrices,
    NonSquareMatrix,
    NotAColumnVector,
    NotARowVector,
    RowsLessThanColumns,
    SingularMatrix,
    NullDataInMatrix,
    OutOfMemory,
};

fn Matrix(comptime T: type) type {
    return struct {
        data: [][]T,
        n: usize,
        p: usize,
        determinant: ?T,
        const Self = @This();
        /// Initialise a matrix
        pub fn init(n: usize, p: usize, allocator: Allocator) !Self {
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
                        identity.data[i][j] = @as(T, 1);
                    } else {
                        identity.data[i][j] = @as(T, 0);
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
        pub fn clone(self: *const Self, allocator: Allocator) !Self {
            var copy = try Matrix(T).init(self.n, self.p, allocator);
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    copy.data[i][j] = self.data[i][j];
                }
            }
            return copy;
        }
        /// Print matrix
        pub fn print(self: Self) !void {
            for (0..self.n) |i| {
                std.debug.print("{d:.2}\n", .{self.data[i]});
            }
        }
        /// Slice matrix
        pub fn slice(self: Self, row_indexes: []const usize, column_indexes: []const usize, allocator: Allocator) !Self {
            var S = try Matrix(T).init(row_indexes.len, column_indexes.len, allocator);
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
            var product = try Matrix(T).init(n, p, allocator);
            for (0..n) |i| {
                for (0..p) |j| {
                    var dot_product: T = self.data[i][0] * b.data[0][j];
                    for (1..self.p) |k| {
                        dot_product += self.data[i][k] * b.data[k][j];
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
            var product = try Matrix(T).init(n, p, allocator);
            for (0..n) |i| {
                for (0..p) |j| {
                    var dot_product: T = self.data[i][0] * b.data[j][0]; // transposed b
                    for (1..self.p) |k| {
                        dot_product += self.data[i][k] * b.data[j][k]; // transposed b
                    }
                    product.data[i][j] = dot_product;
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
            var product = try Matrix(T).init(n, p, allocator);
            for (0..n) |i| {
                for (0..p) |j| {
                    var dot_product: T = self.data[0][i] * b.data[0][j]; // transposed a
                    for (1..self.n) |k| {
                        dot_product += self.data[k][i] * b.data[k][j]; // transposed a
                    }
                    product.data[i][j] = dot_product;
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
            var n = self.n;
            if (self.n > self.p) {
                n = self.p;
            }
            for (row_indexes) |i| {
                var i_max: usize = i;
                for (row_indexes[i..]) |j| {
                    if (@abs(self.data[i_max][i]) < @abs(self.data[j][i])) {
                        i_max = j;
                    }
                }
                row_indexes[i] = i_max;
                row_indexes[i_max] = i;
            }
            return row_indexes;
        }
        /// Forbenius norm
        pub fn norm_forbenius(self: Self) !T {
            var sum = self.data[0][0] - self.data[0][0];
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    sum += @abs(self.data[i][j]) * @abs(self.data[i][j]);
                }
            }
            return std.math.sqrt(sum);
        }
        // Add more element-wise norms at some point, e.g. Max norm
        /// Householder reflection
        pub fn reflect_householder(self: Self, allocator: Allocator) !Self {
            // Make sure self is a column vector
            if (self.p != 1) {
                return MatrixError.NotAColumnVector;
            }
            const n = self.n;
            // Define the Forbenius norm
            const frobenius_norm = try self.norm_forbenius();
            // Define the normal vector from which the input vector, self gets projected into
            var normal_vector = try self.clone(allocator);
            defer normal_vector.deinit(allocator);
            var divisor = self.data[0][0];
            if (self.data[0][0] >= @as(T, 0)) {
                divisor += frobenius_norm;
            } else {
                divisor -= frobenius_norm;
            }
            for (0..n) |i| {
                normal_vector.data[i][0] = self.data[i][0] / divisor;
            }
            normal_vector.data[0][0] = @as(T, 1); // set the origin, i.e. first element as 1
            // std.debug.print("normal_vector:\n", .{});
            // try normal_vector.print();
            // Instantiate the output matrix, H
            const H = try Matrix(T).init_identity(n, allocator);
            // Define the multiplier
            const M = try normal_vector.mult_at(normal_vector, allocator);
            defer M.deinit(allocator);
            const multiplier = @as(T, 2) / M.data[0][0]; // 2 / vT*v
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
                    H.data[i][j] -= multiplier * S.data[i][j];
                }
            }
            // std.debug.print("H:\n", .{});
            // try H.print();
            return H;
        }
        /// Gaussian elimination
        /// Additionally updates the determinant of self
        pub fn gaussian_elimination(self: *Self, b: Self, allocator: Allocator) ![2]Self {
            // // Make sure the matrix is square
            // if (self.n != self.p) {
            //     return MatrixError.NonSquareMatrix;
            // }
            // Make sure the two matrices have the same number of rows
            if (self.n != b.n) {
                return MatrixError.IncompatibleMatrices;
            }
            var determinant = @as(T, 1);
            // Define the pivot
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            std.debug.print("row_indexes={any}\n", .{row_indexes});

            // Instatiate the pivoted reduced row echelon form matrices of self and b
            var self_echelon = try Matrix(T).init(self.n, self.p, allocator);
            var b_echelon = try Matrix(T).init(b.n, b.p, allocator);
            for (row_indexes, 0..) |i_pivot, i| {
                if (i_pivot != i) {
                    determinant = -determinant;
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
                determinant *= a_ii;
                // Set the digonals as one
                for (0..self_echelon.p) |j| {
                    if (self_echelon.data[i][j] != @as(T, 0)) {
                        self_echelon.data[i][j] /= a_ii;
                    }
                }
                for (0..b_echelon.p) |j| {
                    if (b_echelon.data[i][j] != @as(T, 0)) {
                        b_echelon.data[i][j] /= a_ii;
                    }
                }
                if ((i + 1) == self_echelon.n) {
                    break;
                }
                // Subtract the product of the current diagonal value which is one and the value of the value below it to get a zero,
                // and do this for the whole row below it, where the values below the values to the left of the current diagonal value
                // are zero from previous iterations which also render all values below them to zero.
                for (0..(i + 1)) |k| {
                    const a_i_1k = self_echelon.data[i + 1][k];
                    for (0..self_echelon.p) |j| {
                        self_echelon.data[i + 1][j] -= a_i_1k * self_echelon.data[k][j];
                    }
                    for (0..b_echelon.p) |j| {
                        b_echelon.data[i + 1][j] -= a_i_1k * b_echelon.data[k][j];
                    }
                }
                // std.debug.print("Iteration {any}\n", .{i});
                // try self_echelon.print();
            }
            // std.debug.print("determinant={any}\n", .{determinant});
            // std.debug.print("[AFTER]\n", .{});
            // try self_echelon.print();
            if (@abs(determinant) < @as(T, 0.000001)) {
                return MatrixError.SingularMatrix;
            }
            // Update the determinant field of self
            self.determinant = determinant;
            // Reverse: from the lower-right corner to the upper-left corner
            for (0..self_echelon.n) |i_inverse| {
                const i = self_echelon.n - (i_inverse + 1);
                if (i == 0) {
                    break;
                }
                for (0..(i_inverse + 1)) |k_inverse| {
                    const k = self_echelon.n - (k_inverse + 1);
                    const a_i_1k = self_echelon.data[i - 1][k];
                    for (0..self_echelon.p) |j| {
                        self_echelon.data[i - 1][j] -= a_i_1k * self_echelon.data[k][j];
                    }
                    for (0..b_echelon.p) |j| {
                        b_echelon.data[i - 1][j] -= a_i_1k * b_echelon.data[k][j];
                    }
                }
                // std.debug.print("Iteration {any}\n", .{i});
                // try self_echelon.print();
            }
            // std.debug.print("[FINAL self_echelon]\n", .{});
            // try self_echelon.print();
            // std.debug.print("[FINAL b_echelon]\n", .{});
            // try b_echelon.print();
            // Update the determinants of the 2 ouput matrices just for completeness
            self_echelon.determinant = @as(T, 1);
            for (0..self_echelon.n) |i| {
                if (i < self_echelon.p) {
                    self_echelon.determinant.? *= self_echelon.data[i][i];
                }
            }
            b_echelon.determinant = @as(T, 1);
            for (0..b_echelon.n) |i| {
                if (i < b_echelon.p) {
                    b_echelon.determinant.? *= b_echelon.data[i][i];
                }
            }
            return [2]Self{ self_echelon, b_echelon };
        }
        /// LU decomposition (Ref: https://rosettacode.org/wiki/LU_decomposition)
        /// Additionally updates the determinant of self
        pub fn lu(self: *Self, allocator: Allocator) ![3]Self {
            // Make sure the matrix is square
            if (self.n != self.p) {
                return MatrixError.NonSquareMatrix;
            }
            // Instantiate the permutation, L, and U matrices
            var P = try Matrix(T).init(self.n, self.p, allocator);
            var L = try Matrix(T).init(self.n, self.p, allocator);
            var U = try Matrix(T).init(self.n, self.p, allocator);
            // Populate with zeros
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    P.data[i][j] = @as(T, 0);
                    L.data[i][j] = @as(T, 0);
                    U.data[i][j] = @as(T, 0);
                }
            }
            // Define the permutation matrix
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            for (0..self.n) |i| {
                P.data[row_indexes[i]][i] = @as(T, 1);
            }
            // Decompose
            for (0..self.p) |j| {
                L.data[j][j] = @as(T, 1);
                for (row_indexes[0..(j + 1)], 0..) |i_a, i| {
                    var s1 = @as(T, 0);
                    for (0..i) |k| {
                        s1 += U.data[k][j] * L.data[i][k];
                    }
                    U.data[i][j] = self.data[i_a][j] - s1;
                }
                for (row_indexes[j..self.n], j..self.n) |i_a, i| {
                    var s2 = @as(T, 0);
                    for (0..j) |k| {
                        s2 += U.data[k][j] * L.data[i][k];
                    }
                    L.data[i][j] = (self.data[i_a][j] - s2) / U.data[j][j];
                }
            }
            // Append the determinants of each matrix including self, P, L, and U
            P.determinant = @as(T, 1);
            L.determinant = @as(T, 1);
            self.determinant = @as(T, 1);
            for (0..self.n) |i| {
                self.determinant.? *= U.data[i][i];
            }
            return [3]Self{ P, L, U };
        }
        /// QR decomposition (Ref: https://rosettacode.org/wiki/Singular_value_decomposition)
        pub fn qr(self: *Self, allocator: Allocator) ![2]Self {
            // Make sure that self has dimensions n >= p
            if (self.n < self.p) {
                return MatrixError.RowsLessThanColumns;
            }
            const n = self.n;
            var p = self.p;
            if (self.n == self.p) {
                p -= 1;
            }
            var Q = try Matrix(T).init_identity(n, allocator);
            var R = try self.clone(allocator);

            const indexes: []usize = try allocator.alloc(usize, n);
            defer allocator.free(indexes);
            for (0..n) |i| {
                indexes[i] = i;
            }
            std.debug.print("indexes={any}\n", .{indexes});

            for (0..p) |j| {
                std.debug.print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
                std.debug.print("j={any}\n", .{j});
                var H = try Matrix(T).init_identity(n, allocator);
                const a = try R.slice(indexes[j..], indexes[j..(j + 1)], allocator);
                // std.debug.print("a:\n", .{});
                // try a.print();
                const h = try a.reflect_householder(allocator);
                // std.debug.print("h:\n", .{});
                // try h.print();
                for (j..n, 0..) |H_i, h_i| {
                    for (j..n, 0..) |H_j, h_j| {
                        H.data[H_i][H_j] = h.data[h_i][h_j];
                    }
                }
                std.debug.print("H:\n", .{});
                try H.print();
                const Q_mult = try Q.mult(H, allocator);
                defer Q_mult.deinit(allocator);
                // std.debug.print("Q_mult:\n", .{});
                // try Q_mult.print();
                const R_mult = try H.mult(R, allocator);
                defer R_mult.deinit(allocator);
                // std.debug.print("R_mult:\n", .{});
                // try R_mult.print();
                for (0..n) |Q_i| {
                    for (0..n) |Q_j| {
                        Q.data[Q_i][Q_j] = Q_mult.data[Q_i][Q_j];
                        // if (@abs(Q.data[Q_i][Q_j]) < @as(T, 0.00001)) {
                        //     Q.data[Q_i][Q_j] = @as(T, 0);
                        // }
                        if (Q_j < R.p) {
                            R.data[Q_i][Q_j] = R_mult.data[Q_i][Q_j];
                            // if (@abs(R.data[Q_i][Q_j]) < @as(T, 0.00001)) {
                            //     R.data[Q_i][Q_j] = @as(T, 0);
                            // }
                        }
                    }
                }
                std.debug.print("Q:\n", .{});
                try Q.print();
                std.debug.print("R:\n", .{});
                try R.print();
            }

            // std.debug.print("Q:\n", .{});
            // try Q.print();
            // std.debug.print("R:\n", .{});
            // try R.print();
            return [2]Self{ Q, R };
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

test "Pivot, norms, and reflections" {
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
    try expect(@abs(norm_f - 10.29563) < 0.00001);

    //Reflection
    var b = try Matrix(f64).init(p, 1, allocator);
    defer b.deinit(allocator);
    const contents_b = [3]f64{ 12, 6, -4 };
    for (0..n) |i| {
        b.data[i][0] = contents_b[i];
    }

    const H = try b.reflect_householder(allocator);
    std.debug.print("H: {any}\n", .{H});

    const expected_housholder_reflections = [9]f64{ -0.85714286, -0.42857143, 0.28571429, -0.42857143, 0.9010989, 0.06593407, 0.28571429, 0.06593407, 0.95604396 };
    for (0..n) |i| {
        for (0..n) |j| {
            const k = (i * n) + j;
            try expect(@abs(H.data[i][j] - expected_housholder_reflections[k]) < 0.00001);
        }
    }
}

test "Gaussian elimination, decompositions, inverses & determinant" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Gaussian elimination, decompositions, inverses & determinant", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const n: usize = 5;
    const p: usize = 5;
    var a = try Matrix(f64).init(n, p, allocator);
    defer a.deinit(allocator);
    try expect(a.n == n);
    try expect(a.p == p);
    std.debug.print("a.data[0][0]={?}\n", .{a.data[0][0]});
    std.debug.print("a={any}\n", .{a});
    // Populate with data from Example2 in https://rosettacode.org/wiki/LU_decomposition
    var b = try Matrix(f64).init(n, p, allocator);
    defer b.deinit(allocator);
    // const contents = [16]f64{ 11.0, 9.0, 24.0, 2.0, 1.0, 5.0, 2.0, 6.0, 3.0, 17.0, 18.0, 1.0, 2.0, 5.0, 7.0, 1.0 };
    const contents = [25]f64{ 12, 22, 35, 64, 2, 16, 72, 81, 19, 100, 101, 312, 143, 34, 5, 156, 12, 56, 97, 312, 546, 7, 28, 586, 970 };
    // const contents = [9]f64{ 2, 9, 4, 7, 5, 3, 6, 1, 8 };
    // const contents = [16]f64{ 2, 2, 3, 4, 2.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // MatrixError.SingularMatrix
    for (0..n) |i| {
        for (0..p) |j| {
            a.data[i][j] = contents[(i * p) + j];
            b.data[j][i] = contents[(i * p) + j];
        }
    }
    // Identity matrix
    const identity = try Matrix(f64).init_identity(n, allocator);
    defer identity.deinit(allocator);

    std.debug.print("a={any}\n", .{a});
    std.debug.print("b={any}\n", .{b});
    std.debug.print("identity={any}\n", .{identity});

    // Gaussian elimination
    var timer = try std.time.Timer.start();
    const echelons = try a.gaussian_elimination(identity, allocator);
    var time_elapsed = timer.read();
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

    // LU decomposition
    timer.reset();
    time_elapsed = timer.read();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});
    timer.reset();
    const out_test = try a.lu(allocator);
    time_elapsed = timer.read();
    std.debug.print("Time elapsed: {any}\n", .{time_elapsed});

    std.debug.print("a={any}\n", .{a});

    defer out_test[0].deinit(allocator);
    defer out_test[1].deinit(allocator);
    defer out_test[2].deinit(allocator);
    std.debug.print("out_test[0]\n", .{});
    try out_test[0].print();
    std.debug.print("out_test[1]\n", .{});
    try out_test[1].print();
    std.debug.print("out_test[2]\n", .{});
    try out_test[2].print();

    // QR decomposition
    // var A = try Matrix(f32).init(4, 3, allocator);
    // const contents_A = [12]f32{ 12, -51, 4, 6, 167, -68, -4, 24, -41, 10, 2, -7 };
    // for (0..4) |i| {
    //     for (0..3) |j| {
    //         const idx = (i * 3) + j;
    //         A.data[i][j] = contents_A[idx];
    //     }
    // }
    // std.debug.print("A:\n", .{});
    // try A.print();

    // timer.reset();
    // const out_qr = try A.qr(allocator);

    timer.reset();
    const QR = try a.qr(allocator);
    time_elapsed = timer.read();
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
}
