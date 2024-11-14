const std: type = @import("std");
const Allocator: type = std.mem.Allocator;
const linalg: type = @import("linalg.zig");
const expect = std.testing.expect;

/// Error types
const StatsError = error{
    EmptyVector,
    DivisionByZero,
};

pub fn mean(x: []const f64) !f64 {
    const n: usize = x.len;
    if (n < 1) {
        return StatsError.EmptyVector;
    }
    var sum: f64 = 0.0;
    var N: f64 = 0.0;
    for (0..n) |i| {
        sum += x[i];
        N += 1.0;
    }
    return sum / N;
}

pub fn samnple_unif(n: usize, min: f64, max: f64, allocator: Allocator) ![]f64 {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var out: []f64 = try allocator.alloc(f64, n);
    for (0..n) |i| {
        out[i] = (rand.float(f64) * (max - min)) + min;
    }
    return out;
}

/// Marsaglia polar method of sampling from a normal  distribution
pub fn sample_univariate_gaussian(n: usize, mu: f64, sd: f64, allocator: Allocator) ![]f64 {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var out: []f64 = try allocator.alloc(f64, n);
    var u: f64 = 0.0;
    var v: f64 = 0.0;
    var s: f64 = 0.0;
    var z: f64 = 0.0;
    var i: usize = 0;
    while (i < n) {
        // Sample u and v from a uniform distribution between -1 and 1
        u = (rand.float(f64) * 2) - 1;
        v = (rand.float(f64) * 2) - 1;
        s = std.math.pow(f64, u, 2) + std.math.pow(f64, v, 2);
        if (s >= 1.00) {
            continue;
        }
        z = u * @sqrt(-2.00 * @log(s) / s);
        out[i] = (z * sd) + mu;
        i += 1;
        if (i < n) {
            z = v * @sqrt(-2.00 * @log(s) / s);
            out[i] = (z * sd) + mu;
            i += 1;
        }
    }
    return out;
}

/// Probability density function of univariate Gaussian distribution
/// TODO: return a vector
pub fn pdf_univariate_gaussian(y: f64, mu: f64, sd: f64) !f64 {
    if (sd <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const variance: f64 = std.math.pow(f64, sd, 2.00);
    return (1.00 / @sqrt(2.00 * std.math.pi * variance)) *
        @exp(-std.math.pow(f64, (y - mu), 2.00) / (2.00 * variance));
}

/// Numerical approximation of the error function of the standard normal distribution by Yaya Dia in 2023
/// Citation: Dia, Yaya D., Approximate Incomplete Integrals, Application to Complementary Error Function (June 21, 2023). Available at SSRN: https://ssrn.com/abstract=4487559 or http://dx.doi.org/10.2139/ssrn.4487559
/// TODO: return a vector
pub fn cdf_univariate_gaussian(y: f64, mu: f64, sd: f64, lower_tail: bool) !f64 {
    var x: f64 = (y - mu) / sd;
    if (lower_tail) {
        x = y;
    } else {
        x = -y;
    }
    const d0: f64 = x + 2.92678600515804815;
    if (d0 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const d1: f64 = std.math.pow(f64, x, 2.00) + (5.81582518933527391 * x) + 8.97280659046817350;
    if (d1 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const d2: f64 = std.math.pow(f64, x, 2.00) + (5.70347935898051437 * x) + 10.27157061171363079;
    if (d2 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const d3: f64 = std.math.pow(f64, x, 2.00) + (5.51862483025707963 * x) + 12.72323261907760928;
    if (d3 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const d4: f64 = std.math.pow(f64, x, 2.00) + (5.26184239579604207 * x) + 16.88639562007936908;
    if (d4 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const d5: f64 = std.math.pow(f64, x, 2.00) + (4.92081346632882033 * x) + 24.12333774572479110;
    if (d5 <= 0.0) {
        return StatsError.DivisionByZero;
    }
    const phi: f64 = 1.00 - (0.39894228040143268 / d0 *
        (std.math.pow(f64, x, 2.00) + (8.42742300458043240 * x) + 18.38871225773938487) / d1 *
        (std.math.pow(f64, x, 2.00) + (7.30756258553673541 * x) + 18.25323235347346525) / d2 *
        (std.math.pow(f64, x, 2.00) + (5.66479518878470765 * x) + 18.61193318971775795) / d3 *
        (std.math.pow(f64, x, 2.00) + (4.91396098895240075 * x) + 24.14804072812762821) / d4 *
        (std.math.pow(f64, x, 2.00) + (3.83362947800146179 * x) + 11.61511226260603247) / d5 *
        @exp(-(std.math.pow(f64, x, 2.00) / 2)));
    return phi;
}

/// Quantile function
/// Ref: Shore (1982)
/// TODO: return a vector
pub fn quantile_univariate_gaussian(p: f64, mu: f64, sd: f64, lower_tail: bool) !f64 {
    var z: f64 = 0.0;
    if (lower_tail) {
        z = 5.5556 * (1.00 - std.math.pow(f64, (1.00 - p) / p, 0.1186));
    } else {
        z = -5.5556 * (1.00 - std.math.pow(f64, p / (1.00 - p), 0.1186));
    }
    return (z * sd) + mu;
}

// TODO: tests
test "stats" {
    const v = [5]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const mu: f64 = try mean(&v);
    try expect(mu == 3.0);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const x_unif = try samnple_unif(1_000_000, -1.00, 1.00, allocator);
    // std.debug.print("x_unif={any}\n", .{x_unif});
    const mu_x_unif: f64 = try mean(x_unif);
    std.debug.print("mu_x_unif={any}\n", .{mu_x_unif});
    try expect(@abs(mu_x_unif - 0.0) < 0.01);

    const x_gauss = try sample_univariate_gaussian(1_000_000, 3.14, 1.00, allocator);
    // std.debug.print("x_gauss={any}\n", .{x_gauss});
    const mu_x_gauss: f64 = try mean(x_gauss);
    std.debug.print("mu_x_gauss={any}\n", .{mu_x_gauss});
    try expect(@abs(mu_x_gauss - 3.14) < 0.01);

    const d_gauss = try pdf_univariate_gaussian(0.0, 3.14, 1.00);
    std.debug.print("d_gauss={any}\n", .{d_gauss});
    try expect(@abs(d_gauss - 0.002883534) < 0.00001);

    const p0 = try cdf_univariate_gaussian(0.0, 0.0, 1.0, true);
    const p1 = try cdf_univariate_gaussian(0.5, 0.0, 1.0, true);
    const p2 = try cdf_univariate_gaussian(1.0, 0.0, 1.0, true);
    const p3 = try cdf_univariate_gaussian(2.0, 0.0, 1.0, true);
    try expect(@abs(p0 - 0.5000000) < 0.00001);
    try expect(@abs(p1 - 0.6914625) < 0.00001);
    try expect(@abs(p2 - 0.8413447) < 0.00001);
    try expect(@abs(p3 - 0.9772499) < 0.00001);
}
