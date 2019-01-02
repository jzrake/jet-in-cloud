#pragma once
#include <cmath>
#include <algorithm>




// ============================================================================
namespace newtonian_hydro {

    // Indexes to primitive quanitites P
    enum {
        RHO = 0,
        V11 = 1,
        V22 = 2,
        V33 = 3,
        PRE = 4,
    };

    // Indexes to conserved quanitites U
    enum {
        DDD = 0,
        S11 = 1,
        S22 = 2,
        S33 = 3,
        NRG = 4,
    };

    using Vars = std::array<double, 5>;
    using Unit = std::array<double, 3>;
    using Position = std::array<double, 2>;

    struct cons_to_prim;
    struct prim_to_cons;
    struct prim_to_flux;
    struct prim_to_eval;
    struct riemann_hlle;
    struct sph_geom_src_terms;
}




// ============================================================================
struct newtonian_hydro::cons_to_prim
{
    inline Vars operator()(Vars U) const
    {
        const double gm1 = gammaLawIndex - 1.0;
        const double pp = U[S11] * U[S11] + U[S22] * U[S22] + U[S33] * U[S33];
        auto P = Vars();

        P[RHO] =  U[DDD];
        P[PRE] = (U[NRG] - 0.5 * pp / U[DDD]) * gm1;
        P[V11] =  U[S11] / U[DDD];
        P[V22] =  U[S22] / U[DDD];
        P[V33] =  U[S33] / U[DDD];

        return P;
    }
    double gammaLawIndex = 5. / 3;
};




// ============================================================================
struct newtonian_hydro::prim_to_cons
{
    inline Vars operator()(Vars P) const
    {
        const double gm1 = gammaLawIndex - 1.0;
        const double vv = P[V11] * P[V11] + P[V22] * P[V22] + P[V33] * P[V33];
        auto U = Vars();

        U[DDD] = P[RHO];
        U[S11] = P[RHO] * P[V11];
        U[S22] = P[RHO] * P[V22];
        U[S33] = P[RHO] * P[V33];
        U[NRG] = P[RHO] * 0.5 * vv + P[PRE] / gm1;

        return U;
    }
    double gammaLawIndex = 5. / 3;
};




// ============================================================================
struct newtonian_hydro::prim_to_flux
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double vn = P[V11] * N[0] + P[V22] * N[1] + P[V33] * N[2];
        auto U = prim_to_cons()(P);
        auto F = Vars();

        F[DDD] = vn * U[DDD];
        F[S11] = vn * U[S11] + P[PRE] * N[0];
        F[S22] = vn * U[S22] + P[PRE] * N[1];
        F[S33] = vn * U[S33] + P[PRE] * N[2];
        F[NRG] = vn * U[NRG] + P[PRE] * vn;

        return F;
    }
    double gammaLawIndex = 5. / 3;
};




// ============================================================================
struct newtonian_hydro::prim_to_eval
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double gm0 = gammaLawIndex;
        const double cs = std::sqrt(gm0 * P[PRE] / P[RHO]);
        const double vn = P[V11] * N[0] + P[V22] * N[1] + P[V33] * N[2];
        auto A = Vars();

        A[0] = vn - cs;
        A[1] = vn;
        A[2] = vn;
        A[3] = vn;
        A[4] = vn + cs;

        return A;
    }
    double gammaLawIndex = 5. / 3;
};




// ============================================================================
struct newtonian_hydro::riemann_hlle
{
    riemann_hlle(Unit nhat) : nhat(nhat) {}

    inline Vars operator()(Vars Pl, Vars Pr) const
    {
        auto Ul = p2c(Pl);
        auto Ur = p2c(Pr);
        auto Al = p2a(Pl, nhat);
        auto Ar = p2a(Pr, nhat);
        auto Fl = p2f(Pl, nhat);
        auto Fr = p2f(Pr, nhat);

        const double epl = *std::max_element(Al.begin(), Al.end());
        const double eml = *std::min_element(Al.begin(), Al.end());
        const double epr = *std::max_element(Ar.begin(), Ar.end());
        const double emr = *std::min_element(Ar.begin(), Ar.end());
        const double ap = std::max(0.0, std::max(epl, epr));
        const double am = std::min(0.0, std::min(eml, emr));

        Vars U, F;

        for (int q = 0; q < 5; ++q)
        {
            U[q] = (ap * Ur[q] - am * Ul[q] + (Fl[q] - Fr[q])) / (ap - am);
            F[q] = (ap * Fl[q] - am * Fr[q] - (Ul[q] - Ur[q]) * ap * am) / (ap - am);
        }
        return F;
    }
    Unit nhat;
    prim_to_cons p2c;
    prim_to_eval p2a;
    prim_to_flux p2f;
};




// ============================================================================
struct newtonian_hydro::sph_geom_src_terms
{
    inline Vars operator()(Vars P, Position X) const
    {
        const double r = X[0];
        const double q = X[1];
        const double dg = P[0];
        const double vr = P[1];
        const double vq = P[2];
        const double vp = P[3];
        const double pg = P[4];
        auto S = Vars();

        S[0] = 0.0;
        S[1] = (2 * pg + dg * (vq * vq + vp * vp)) / r;
        S[2] = (pg * cot(q) + dg * (vp * vp * cot(q) - vr * vq)) / r;
        S[3] = -dg * vp * (vr + vq * cot(q)) / r;
        S[4] = 0.0;

        return S;
    }
    double cot(double x) const
    {
        return std::tan(M_PI_2 - x);
    }
};




// ============================================================================
namespace sr_hydro {

    // Indexes to primitive quanitites P
    enum {
        RHO = 0,
        V11 = 1,
        V22 = 2,
        V33 = 3,
        PRE = 4,
    };

    // Indexes to conserved quanitites U
    enum {
        DDD = 0,
        S11 = 1,
        S22 = 2,
        S33 = 3,
        TAU = 4,
    };

    using Vars = std::array<double, 5>;
    using Unit = std::array<double, 3>;
    using Position = std::array<double, 2>;

    struct cons_to_prim;
    struct prim_to_cons;
    struct prim_to_flux;
    struct prim_to_eval;
    struct riemann_hlle;
    struct sph_geom_src_terms;

    std::string to_string(Vars);
    std::string to_string(Position);
}




// ============================================================================
struct sr_hydro::cons_to_prim
{
    inline Vars operator()(Vars U, Position X) const
    {
        const double gm  = gammaLawIndex;
        const double D   = U[DDD];
        const double tau = U[TAU];
        const double SS  = U[S11] * U[S11] + U[S22] * U[S22] + U[S33] * U[S33];

        int soln_found = 0;
        int n_iter = 0;
        double f;
        double g;
        double W0 = 1.0;
        double p = 1.0; // guess pressure

        while (! soln_found)
        {
            double v2  = SS / std::pow(tau + D + p, 2);
            double W2  = 1.0 / (1.0 - v2);
            double W   = std::sqrt(W2);
            double e   = (tau + D * (1.0 - W) + p * (1.0 - W2)) / (D * W);
            double d   = D / W;
            double h   = 1.0 + e + p / d;
            double cs2 = gm * p / (d * h);

            f = d * e * (gm - 1.0) - p;
            g = v2 * cs2 - 1.0;
            p -= f / g;

            if (std::fabs(f) < errorTolerance)
            {
                W0 = W;
                soln_found = 1;
            }
            if (++n_iter == newtonIterMax)
            {
                throw std::invalid_argument("c2p failure: root finder not converging U="
                    + to_string(U) + " at X=" + to_string(X));
            }
        }

        auto P = Vars();

        P[RHO] = D / W0;
        P[PRE] = p;
        P[V11] = U[S11] / (tau + D + p);
        P[V22] = U[S22] / (tau + D + p);
        P[V33] = U[S33] / (tau + D + p);

        if (P[PRE] < 0.0 && ! allowNegativePressure)
        {
            throw std::invalid_argument("c2p failure: negative pressure U=" + to_string(U));
        }
        if (P[RHO] < 0.0)
        {
            throw std::invalid_argument("c2p failure: negative density U=" + to_string(U));
        }
        if (std::isnan(W0))
        {
            throw std::invalid_argument("c2p failure: nan W U=" + to_string(U));
        }
        return P;
    }

    const int newtonIterMax = 50;
    const double errorTolerance = 1e-10;
    const double maxW = 1e12;
    const double gammaLawIndex = 4. / 3;
    const bool allowNegativePressure = true;
};




// ============================================================================
struct sr_hydro::prim_to_cons
{
    inline Vars operator()(Vars P) const
    {
        const double gm1 = gammaLawIndex - 1.0;
        const double v2 = P[V11] * P[V11] + P[V22] * P[V22] + P[V33] * P[V33];
        const double W2 = 1.0 / (1.0 - v2);
        const double W = std::sqrt(W2);
        const double e = P[PRE] / (P[RHO] * gm1);
        const double h = 1.0 + e + P[PRE] / P[RHO];
        auto U = Vars();

        U[DDD] = P[RHO] * W;
        U[S11] = P[RHO] * h * W2 * P[V11];
        U[S22] = P[RHO] * h * W2 * P[V22];
        U[S33] = P[RHO] * h * W2 * P[V33];
        U[TAU] = P[RHO] * h * W2 - P[PRE] - U[DDD];

        return U;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sr_hydro::prim_to_flux
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double vn = P[V11] * N[0] + P[V22] * N[1] + P[V33] * N[2];
        auto U = prim_to_cons()(P);
        auto F = Vars();

        F[DDD] = vn * U[DDD];
        F[S11] = vn * U[S11] + P[PRE] * N[0];
        F[S22] = vn * U[S22] + P[PRE] * N[1];
        F[S33] = vn * U[S33] + P[PRE] * N[2];
        F[TAU] = vn * U[TAU] + P[PRE] * vn;

        return F;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sr_hydro::prim_to_eval
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double gm0 = gammaLawIndex;
        const double gm1 = gammaLawIndex - 1.0;
        const double d   = P[RHO];
        const double p   = std::max(0.0, P[PRE]);
        const double e   = p / (d * gm1);
        const double h   = 1.0 + e + p / d;
        const double cs2 = gm0 * p / (d * h);
        const double vn  = P[V11] * N[0]   + P[V22] * N[1]   + P[V33] * N[2];
        const double vv  = P[V11] * P[V11] + P[V22] * P[V22] + P[V33] * P[V33];
        const double v2  = vn * vn;
        const double K   = std::sqrt(cs2 * (1 - vv) * (1 - vv * cs2 - v2 * (1 - cs2)));
        auto A = Vars();

        A[0] = (vn * (1 - cs2) - K) / (1 - vv * cs2);
        A[1] = vn;
        A[2] = vn;
        A[3] = vn;
        A[4] = (vn * (1 - cs2) + K) / (1 - vv * cs2);

        return A;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sr_hydro::riemann_hlle
{
    riemann_hlle(Unit nhat) : nhat(nhat) {}

    inline Vars operator()(Vars Pl, Vars Pr) const
    {
        auto Ul = p2c(Pl);
        auto Ur = p2c(Pr);
        auto Al = p2a(Pl, nhat);
        auto Ar = p2a(Pr, nhat);
        auto Fl = p2f(Pl, nhat);
        auto Fr = p2f(Pr, nhat);

        const double epl = *std::max_element(Al.begin(), Al.end());
        const double eml = *std::min_element(Al.begin(), Al.end());
        const double epr = *std::max_element(Ar.begin(), Ar.end());
        const double emr = *std::min_element(Ar.begin(), Ar.end());
        const double ap = std::max(0.0, std::max(epl, epr));
        const double am = std::min(0.0, std::min(eml, emr));

        Vars U, F;

        for (int q = 0; q < 5; ++q)
        {
            U[q] = (ap * Ur[q] - am * Ul[q] + (Fl[q] - Fr[q])) / (ap - am);
            F[q] = (ap * Fl[q] - am * Fr[q] - (Ul[q] - Ur[q]) * ap * am) / (ap - am);
        }
        return F;
    }
    Unit nhat;
    prim_to_cons p2c;
    prim_to_eval p2a;
    prim_to_flux p2f;
};




// ============================================================================
struct sr_hydro::sph_geom_src_terms
{
    inline Vars operator()(Vars P, Position X) const
    {
        const double r = X[0];
        const double q = X[1];
        const double gm = gammaLawIndex;
        const double dg = P[RHO];
        const double vr = P[V11];
        const double vq = P[V22];
        const double vp = P[V33];
        const double pg = P[PRE];
        const double eg = pg / dg / (gm - 1);
        const double v2 = vr * vr + vq * vq + vp * vp;
        const double W2 = 1.0 / (1.0 - v2);
        const double hg = 1.0 + eg + pg / dg;
        const double rhohW2 = dg * hg * W2;
        auto S = Vars();

        S[0] = 0.0;
        S[1] = (2 * pg + rhohW2 * (vq * vq + vp * vp)) / r;
        S[2] = (pg * cot(q) + rhohW2 * (vp * vp * cot(q) - vr * vq)) / r;
        S[3] = -rhohW2 * vp * (vr + vq * cot(q)) / r;
        S[4] = 0.0;

        return S;
    }
    double cot(double x) const
    {
        return std::tan(M_PI_2 - x);
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
std::string sr_hydro::to_string(sr_hydro::Vars V)
{
    char res[1024];
    std::snprintf(res, 1024, "[%4.3e %4.3e %4.3e %4.3e %4.3e]", V[0], V[1], V[2], V[3], V[4]);
    return res;
}

std::string sr_hydro::to_string(sr_hydro::Position X)
{
    char res[1024];
    std::snprintf(res, 1024, "[r=%f q=%f]", X[0], X[1]);
    return res;
}




// ============================================================================
namespace sru_hydro {

    // Indexes to primitive quanitites P
    enum {
        RHO = 0,
        U11 = 1,
        U22 = 2,
        U33 = 3,
        PRE = 4,
    };

    // Indexes to conserved quanitites U
    enum {
        DDD = 0,
        S11 = 1,
        S22 = 2,
        S33 = 3,
        TAU = 4,
    };

    using Vars = std::array<double, 5>;
    using Unit = std::array<double, 3>;
    using Position = std::array<double, 2>;

    struct cons_to_prim;
    struct prim_to_cons;
    struct prim_to_flux;
    struct prim_to_eval;
    struct riemann_hlle;
    struct sph_geom_src_terms;

    std::string to_string(Vars);
    std::string to_string(Position);
}




// ============================================================================
struct sru_hydro::cons_to_prim
{
    inline Vars operator()(Vars U, Position X) const
    {
        const double gm  = gammaLawIndex;
        const double D   = U[DDD];
        const double tau = U[TAU];
        const double SS  = U[S11] * U[S11] + U[S22] * U[S22] + U[S33] * U[S33];

        int soln_found = 0;
        int n_iter = 0;
        double f;
        double g;
        double W0 = 1.0;
        double p = 0.0; // guess pressure

        while (! soln_found)
        {
            double v2  = std::min(SS / std::pow(tau + D + p, 2), 1.0 - 1e-10);
            double W2  = 1.0 / (1.0 - v2);
            double W   = std::sqrt(W2);
            double e   = (tau + D * (1.0 - W) + p * (1.0 - W2)) / (D * W);
            double d   = D / W;
            double h   = 1.0 + e + p / d;
            double cs2 = gm * p / (d * h);

            f = d * e * (gm - 1.0) - p;
            g = v2 * cs2 - 1.0;
            p -= f / g;

            if (std::fabs(f) < errorTolerance)
            {
                W0 = W;
                soln_found = 1;
            }
            if (++n_iter == newtonIterMax)
            {
                throw std::invalid_argument("c2p failure: "
                    "root finder not converging\n"
                    "U=" + to_string(U) + "\n"
                    "X=" + to_string(X) + "\n"
                    "error=" + std::to_string(f));
            }
        }

        auto P = Vars();

        P[RHO] = D / W0;
        P[PRE] = p;
        P[U11] = W0 * U[S11] / (tau + D + p);
        P[U22] = W0 * U[S22] / (tau + D + p);
        P[U33] = W0 * U[S33] / (tau + D + p);

        if (P[PRE] < 0.0 && ! allowNegativePressure)
        {
            throw std::invalid_argument("c2p failure: negative pressure U=" + to_string(U));
        }
        if (P[RHO] < 0.0)
        {
            throw std::invalid_argument("c2p failure: negative density U=" + to_string(U));
        }
        if (std::isnan(W0))
        {
            throw std::invalid_argument("c2p failure: nan W U=" + to_string(U));
        }
        return P;
    }

    const int newtonIterMax = 50;
    const double errorTolerance = 1e-10;
    const double maxW = 1e12;
    const double gammaLawIndex = 4. / 3;
    const bool allowNegativePressure = true;
};




// ============================================================================
struct sru_hydro::prim_to_cons
{
    inline Vars operator()(Vars P) const
    {
        const double gm1 = gammaLawIndex - 1.0;
        const double u2 = P[U11] * P[U11] + P[U22] * P[U22] + P[U33] * P[U33];
        const double W = std::sqrt(1.0 + u2);
        const double e = P[PRE] / (P[RHO] * gm1);
        const double h = 1.0 + e + P[PRE] / P[RHO];
        auto U = Vars();

        U[DDD] = P[RHO] * W;
        U[S11] = U[DDD] * P[U11] * h;
        U[S22] = U[DDD] * P[U22] * h;
        U[S33] = U[DDD] * P[U33] * h;
        U[TAU] = U[DDD] * W * h - P[PRE] - U[DDD];

        return U;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sru_hydro::prim_to_flux
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double uu = P[U11] * P[U11] + P[U22] * P[U22] + P[U33] * P[U33];
        const double vn = (P[U11] * N[0] + P[U22] * N[1] + P[U33] * N[2]) / std::sqrt(1 + uu);

        auto U = prim_to_cons()(P);
        auto F = Vars();

        F[DDD] = vn * U[DDD];
        F[S11] = vn * U[S11] + P[PRE] * N[0];
        F[S22] = vn * U[S22] + P[PRE] * N[1];
        F[S33] = vn * U[S33] + P[PRE] * N[2];
        F[TAU] = vn * U[TAU] + P[PRE] * vn;

        return F;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sru_hydro::prim_to_eval
{
    inline Vars operator()(Vars P, Unit N) const
    {
        const double gm0 = gammaLawIndex;
        const double gm1 = gammaLawIndex - 1.0;
        const double uu  = P[U11] * P[U11] + P[U22] * P[U22] + P[U33] * P[U33];
        const double vv  = uu / (1 + uu);
        const double vn  = (P[U11] * N[0] + P[U22] * N[1] + P[U33] * N[2]) / std::sqrt(1 + uu);
        const double d   = P[RHO];
        const double p   = std::max(0.0, P[PRE]);
        const double e   = p / (d * gm1);
        const double h   = 1.0 + e + p / d;
        const double cs2 = gm0 * p / (d * h);
        const double v2  = vn * vn;
        const double K   = std::sqrt(cs2 * (1 - vv) * (1 - vv * cs2 - v2 * (1 - cs2)));
        auto A = Vars();

        A[0] = (vn * (1 - cs2) - K) / (1 - vv * cs2);
        A[1] = vn;
        A[2] = vn;
        A[3] = vn;
        A[4] = (vn * (1 - cs2) + K) / (1 - vv * cs2);

        return A;
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
struct sru_hydro::riemann_hlle
{
    riemann_hlle(Unit nhat) : nhat(nhat) {}

    inline Vars operator()(Vars Pl, Vars Pr) const
    {
        auto Ul = p2c(Pl);
        auto Ur = p2c(Pr);
        auto Al = p2a(Pl, nhat);
        auto Ar = p2a(Pr, nhat);
        auto Fl = p2f(Pl, nhat);
        auto Fr = p2f(Pr, nhat);

        const double epl = *std::max_element(Al.begin(), Al.end());
        const double eml = *std::min_element(Al.begin(), Al.end());
        const double epr = *std::max_element(Ar.begin(), Ar.end());
        const double emr = *std::min_element(Ar.begin(), Ar.end());
        const double ap = std::max(0.0, std::max(epl, epr));
        const double am = std::min(0.0, std::min(eml, emr));

        Vars U, F;

        for (int q = 0; q < 5; ++q)
        {
            U[q] = (ap * Ur[q] - am * Ul[q] + (Fl[q] - Fr[q])) / (ap - am);
            F[q] = (ap * Fl[q] - am * Fr[q] - (Ul[q] - Ur[q]) * ap * am) / (ap - am);
        }
        return F;
    }
    Unit nhat;
    prim_to_cons p2c;
    prim_to_eval p2a;
    prim_to_flux p2f;
};




// ============================================================================
struct sru_hydro::sph_geom_src_terms
{
    inline Vars operator()(Vars P, Position X) const
    {
        const double r = X[0];
        const double q = X[1];
        const double gm = gammaLawIndex;
        const double dg = P[RHO];
        const double ur = P[U11];
        const double uq = P[U22];
        const double up = P[U33];
        const double pg = P[PRE];
        const double eg = pg / dg / (gm - 1);
        const double hg = 1.0 + eg + pg / dg;
        auto S = Vars();

        S[0] = 0.0;
        S[1] = (2      * pg + dg * hg * (uq * uq          + up * up)) / r;
        S[2] = (cot(q) * pg + dg * hg * (up * up * cot(q) - ur * uq)) / r;
        S[3] = -dg * hg * up * (ur + uq * cot(q)) / r;
        S[4] = 0.0;

        return S;
    }
    double cot(double x) const
    {
        return std::tan(M_PI_2 - x);
    }
    double gammaLawIndex = 4. / 3;
};




// ============================================================================
std::string sru_hydro::to_string(sru_hydro::Vars V)
{
    char res[1024];
    std::snprintf(res, 1024, "[%4.3e %4.3e %4.3e %4.3e %4.3e]",
        V[0], V[1], V[2], V[3], V[4]);
    return res;
}

std::string sru_hydro::to_string(sru_hydro::Position X)
{
    char res[1024];
    std::snprintf(res, 1024, "[r=%f q=%f]", X[0], X[1]);
    return res;
}