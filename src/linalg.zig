const std: type = @import("std");
const expect = std.testing.expect;
const Allocator: type = std.mem.Allocator;

const MatrixError = error{
    IncompatibleMatrices,
    NonSquareMatrix,
    SingularMatrix,
    NullDataInMatrix,
    OutOfMemory,
};

fn Matrix(comptime T: type) type {
    return struct {
        data: [][]T,
        n: usize,
        p: usize,
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
            };
        }
        /// De-initialise a matrix
        pub fn deinit(self: Self, allocator: Allocator) void {
            for (self.data) |inner| {
                allocator.free(inner);
            }
            allocator.free(self.data);
        }
        // Define constants: zero and one with the correct type
        pub fn define_constants(self: Self) ![2]T {
            const zero = self.data[0][0] - self.data[0][0];
            var one = self.data[0][0] / self.data[0][0];
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    if (self.data[i][j] > zero) {
                        one = self.data[i][j] / self.data[i][j];
                        break;
                    }
                }
            }
            return [2]T{ zero, one };
        }
        /// Initiliase an identity matrix
        pub fn init_identity(n: usize, allocator: Allocator) !Self {
            var identity = try Matrix(T).init(n, n, allocator);
            const constants = try identity.define_constants();
            const zero = constants[0];
            const one = constants[1];
            for (0..n) |i| {
                for (0..n) |j| {
                    if (i == j) {
                        identity.data[i][j] = one;
                    } else {
                        identity.data[i][j] = zero;
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
        /// Initiliase a matrix filled with zeros
        /// Matrix multiplication
        pub fn mult(self: Self, x: Self, allocator: Allocator) !Self {
            if ((self.p != x.n) or (self.n != x.p)) {
                return MatrixError.IncompatibleMatrices;
            }
            const n: usize = self.n;
            const p: usize = x.p;
            var product = try Matrix(T).init(n, p, allocator);
            for (0..n) |i| {
                for (0..p) |j| {
                    var dot_product: T = self.data[i][0] * x.data[0][j];
                    for (1..self.p) |k| {
                        dot_product += self.data[i][k] * x.data[k][j];
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
            // Sort the row indexes by the largest unselected  value per column
            // We will iterate across rows using the column indexes.
            // This is because we want to sort the rows according to which column they correspond to the row with the largest value.
            for (0..self.p) |j| {
                if (j == self.n) {
                    break;
                }
                // We are setting the sorted index of each row consecutively as iterated over by j, i.e. the jth column.
                // We therefore skip the previous rows, i.e. setting i_max to the current column index, j.
                // We iterate across row indexes which are undergoing sorting while skipping the previously sorted rows.
                var i_max: usize = j;
                for (row_indexes[j..]) |i| {
                    if (self.data[i_max][j] < self.data[i][j]) {
                        i_max = i;
                    }
                }
                // Continue iterating across columns if the sorting index of the row is correct.
                if (j == i_max) {
                    continue;
                }
                // Swap the index of current/incorrect row index with the correct/sorting index
                const x0 = row_indexes[j];
                const x1 = row_indexes[i_max];
                row_indexes[j] = x1;
                row_indexes[i_max] = x0;
            }
            return row_indexes;
        }

        // // Gaussian elimination
        // pub fn gaussian_elimination(self: Self, b: Self, allocator: Allocator) !Self {
        //     return self;
        // }

        // LU decomposition (Ref: https://rosettacode.org/wiki/LU_decomposition)
        pub fn lu(self: Self, allocator: Allocator) ![3]Self {
            // Make sure the matrix is square
            if (self.n != self.p) {
                return MatrixError.NonSquareMatrix;
            }
            // Instantiate the permutation, L, and U matrices
            var P = try Matrix(T).init(self.n, self.p, allocator);
            var L = try Matrix(T).init(self.n, self.p, allocator);
            var U = try Matrix(T).init(self.n, self.p, allocator);
            // Constants
            const constants = try self.define_constants();
            const zero = constants[0];
            const one = constants[1];
            // Populate with zeros
            for (0..self.n) |i| {
                for (0..self.p) |j| {
                    P.data[i][j] = zero;
                    L.data[i][j] = zero;
                    U.data[i][j] = zero;
                }
            }
            // Define the permutation matrix
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            for (0..self.n) |i| {
                P.data[row_indexes[i]][i] = one;
            }
            // Decompose
            for (0..self.p) |j| {
                L.data[j][j] = one;
                for (row_indexes[0..(j + 1)], 0..) |i_a, i| {
                    var s1 = zero;
                    for (0..i) |k| {
                        s1 += U.data[k][j] * L.data[i][k];
                    }
                    U.data[i][j] = self.data[i_a][j] - s1;
                }
                for (row_indexes[j..self.n], j..self.n) |i_a, i| {
                    var s2 = zero;
                    for (0..j) |k| {
                        s2 += U.data[k][j] * L.data[i][k];
                    }
                    L.data[i][j] = (self.data[i_a][j] - s2) / U.data[j][j];
                }
            }
            return [3]Self{ P, L, U };
        }

        // Determinant
        pub fn det(self: Self, allocator: Allocator) !T {
            // LU decomposition
            const P_L_U = try self.lu(allocator);
            // Find the determinant as the product of the diagonal elements of U since the diagonals of L are all one.
            var determinant = P_L_U[1].data[0][0];
            for (0..self.n) |i| {
                determinant *= P_L_U[2].data[i][i];
            }
            std.debug.print("determinant={any}\n", .{determinant});
            return determinant;
        }
    };
}

test "linalg" {
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
    std.debug.print("a={any}\n", .{a});
    std.debug.print("b={any}\n", .{b});

    // Define constants one and zero with the correct type
    const constants = try a.define_constants();
    try expect(constants[0] == 0.0);
    try expect(constants[1] == 1.0);

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

    // Determinant
    const determinant = try a.det(allocator);
    try expect(determinant - 284.0 < 0.0001);

    // Matrix multiplication
    const c = try a.mult(b, allocator);
    defer c.deinit(allocator);
    std.debug.print("c={any}\n", .{c});
    const expected_matrix_product = [16]f64{
        782.0, 116.0, 620.0, 237.0,
        116.0, 66.0,  130.0, 47.0,
        620.0, 130.0, 623.0, 218.0,
        237.0, 47.0,  218.0, 79.0,
    };
    for (0..n) |i| {
        for (0..p) |j| {
            const idx: usize = (i * n) + j;
            try expect(c.data[i][j] == expected_matrix_product[idx]);
        }
    }
    // Pivot
    const row_indexes = try a.pivot(allocator);
    defer allocator.free(row_indexes);
    std.debug.print("row_indexes={any}\n", .{row_indexes});
    const expected_row_indexes = [4]usize{ 0, 2, 1, 3 };
    for (0..n) |i| {
        try expect(row_indexes[i] == expected_row_indexes[i]);
    }
    // LU decomposition
    const out_test = try a.lu(allocator);
    std.debug.print("out_test[0]={any}\n", .{out_test[0]});
    std.debug.print("out_test[1]={any}\n", .{out_test[1]});
    std.debug.print("out_test[2]={any}\n", .{out_test[2]});
}
