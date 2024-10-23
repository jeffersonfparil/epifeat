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
        /// Define constants: zero and one with the correct type
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
        /// Gaussian elimination
        pub fn gaussian_elimination(self: Self, b: Self, allocator: Allocator) ![2]Self {
            // // Make sure the matrix is square
            // if (self.n != self.p) {
            //     return MatrixError.NonSquareMatrix;
            // }
            // Make sure the two matrices have the same number of rows
            if (self.n != b.n) {
                return MatrixError.IncompatibleMatrices;
            }
            // Define constants
            // const constants = try self.define_constants();
            // const zero = constants[0];
            // const one = constants[1];
            // var determinant = one;
            // Define the pivot
            const row_indexes = try self.pivot(allocator);
            defer allocator.free(row_indexes);
            std.debug.print("row_indexes={any}\n", .{row_indexes});

            // Instatiate the pivoted reduced row echelon form matrices of self and b
            var self_echelon = try Matrix(T).init(self.n, self.p, allocator);
            var b_echelon = try Matrix(T).init(b.n, b.p, allocator);
            for (row_indexes, 0..) |i_pivot, i| {
                for (0..self.p) |j| {
                    self_echelon.data[i][j] = self.data[i_pivot][j];
                }
                for (0..b.p) |j| {
                    b_echelon.data[i][j] = b.data[i_pivot][j];
                }
            }
            // Perform elementary row operations to convert self into an upper-triangular matrix, with diagonal of ones
            std.debug.print("[BEFORE] self_echelon={any}\n", .{self_echelon});
            // Forward: from the upper-left corner to the lower-right corner
            for (0..self_echelon.n) |i| {
                const a_ii = self_echelon.data[i][i];
                for (0..self_echelon.p) |j| {
                    self_echelon.data[i][j] /= a_ii;
                }
                for (0..b_echelon.p) |j| {
                    b_echelon.data[i][j] /= a_ii;
                }
                if ((i + 1) == self_echelon.n) {
                    break;
                }
                for (0..(i + 1)) |k| {
                    const a_i_1k = self_echelon.data[i + 1][k];
                    for (0..self_echelon.p) |j| {
                        self_echelon.data[i + 1][j] -= a_i_1k * self_echelon.data[k][j];
                    }
                    for (0..b_echelon.p) |j| {
                        b_echelon.data[i + 1][j] -= a_i_1k * b_echelon.data[k][j];
                    }
                }
                std.debug.print("self_echelon-iter={any}={any}\n", .{ i, self_echelon });
            }
            std.debug.print("[AFTER] self_echelon={any}\n", .{self_echelon});

            // Reverse: from the lower-right corner to the upper-left corner
            for (0..self_echelon.n) |i_inverse| {
                const i = self_echelon.n - (i_inverse + 1);
                const a_ii = self_echelon.data[i][i];
                for (0..self_echelon.p) |j| {
                    self_echelon.data[i][j] /= a_ii;
                }
                for (0..b_echelon.p) |j| {
                    b_echelon.data[i][j] /= a_ii;
                }
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
                std.debug.print("self_echelon-REVERSE-iter={any}={any}\n", .{ i, self_echelon });
            }

            // Reverse the pivot by storing output in self_echelon for memory efficiency.
            for (row_indexes, 0..) |i_pivot, i| {
                for (0..b.p) |j| {
                    self_echelon.data[i_pivot][j] = b_echelon.data[i][j];
                }
            }

            std.debug.print("[FINAL] self_echelon={any}\n", .{self_echelon});
            std.debug.print("[FINAL] b_echelon={any}\n", .{b_echelon});

            // return [2]Self{ self_echelon, b_echelon };
            return [2]Self{ b_echelon, self_echelon };
        }
        /// LU decomposition (Ref: https://rosettacode.org/wiki/LU_decomposition)
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

        // // Determinant
        // pub fn det(self: Self, allocator: Allocator) !T {
        //     // LU decomposition
        //     const P_L_U = try self.lu(allocator);
        //     // Find the determinant as the product of the diagonal elements of U since the diagonals of L are all one.
        //     var determinant = P_L_U[1].data[0][0];
        //     for (0..self.n) |i| {
        //         determinant *= P_L_U[2].data[i][i];
        //     }
        //     std.debug.print("determinant={any}\n", .{determinant});
        //     return determinant;
        // }
    };
}

test "Initialisations & cloning" {
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

    // Clone
    const a_copy = try a.clone(allocator);
    defer a_copy.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try expect(a_copy.data[i][j] == a.data[i][j]);
        }
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

test "Gaussian elimination, decompositions, inverses & determinant" {
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
    std.debug.print("Gaussian elimination, decompositions, inverses & determinant", .{});
    std.debug.print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n", .{});
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
    // const contents = [16]f64{ 11.0, 9.0, 24.0, 2.0, 1.0, 5.0, 2.0, 6.0, 3.0, 17.0, 18.0, 1.0, 2.0, 5.0, 7.0, 1.0 };
    const contents = [16]f64{ 2, 2, 3, 4, 2.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
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

    // Pivot
    // const row_indexes = try a.pivot(allocator);
    // defer allocator.free(row_indexes);
    // std.debug.print("row_indexes={any}\n", .{row_indexes});
    // const expected_row_indexes = [4]usize{ 0, 2, 1, 3 };
    // for (0..n) |i| {
    //     try expect(row_indexes[i] == expected_row_indexes[i]);
    // }

    // Gaussian elimination
    const echelons = try a.gaussian_elimination(identity, allocator);
    defer echelons[0].deinit(allocator);
    defer echelons[1].deinit(allocator);
    const a_inverse = echelons[1];
    std.debug.print("echelons[0]={any}\n", .{echelons[0]});
    std.debug.print("a_inverse={any}\n", .{a_inverse});
    const should_be_identity = try a.mult(a_inverse, allocator);
    defer should_be_identity.deinit(allocator);
    std.debug.print("should_be_identity={any}\n", .{should_be_identity});
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

    var det: f64 = 1.00;
    for (0..a.n) |i| {
        det *= echelons[0].data[i][i];
    }
    std.debug.print("det={any}\n", .{det});

    var non_square_matrix = try Matrix(f64).init(5, 3, allocator);
    defer non_square_matrix.deinit(allocator);
    var identity_for_non_square_matrix = try Matrix(f64).init_identity(5, allocator);
    defer identity_for_non_square_matrix.deinit(allocator);
    var counter: usize = 0;
    for (0..5) |i| {
        for (0..3) |j| {
            non_square_matrix.data[i][j] = contents[counter];
            counter += 1;
        }
    }
    const x = try non_square_matrix.gaussian_elimination(identity_for_non_square_matrix, allocator);
    defer x[0].deinit(allocator);
    defer x[1].deinit(allocator);
    std.debug.print("x[0]={any}\n", .{x[0]});
    std.debug.print("x[1]={any}\n", .{x[1]});

    const should_be_identity_non_square = try x[1].mult(non_square_matrix, allocator);
    defer should_be_identity_non_square.deinit(allocator);
    std.debug.print("should_be_identity_non_square={any}\n", .{should_be_identity_non_square});

    // LU decomposition
    const out_test = try a.lu(allocator);
    defer out_test[0].deinit(allocator);
    defer out_test[1].deinit(allocator);
    defer out_test[2].deinit(allocator);
    std.debug.print("out_test[0]={any}\n", .{out_test[0]});
    std.debug.print("out_test[1]={any}\n", .{out_test[1]});
    std.debug.print("out_test[2]={any}\n", .{out_test[2]});
}
