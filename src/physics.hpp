#pragma once
#include <cmath>




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
}




// ============================================================================
struct sr_hydro::cons_to_prim
{
    inline Vars operator()(Vars U) const
    {
        const double gm  = gammaLawIndex;
        const double D   = U[DDD];
        const double Tau = U[TAU];
        const double S2  = U[S11] * U[S11] + U[S22] * U[S22] + U[S33] * U[S33];

        int soln_found = 0;
        int n_iter = 0;
        double f;
        double g;
        double W_soln = 1.0;
        double p = 1.0; // guess pressure

        while (! soln_found)
        {
            double v2  = S2 / std::pow(Tau + D + p, 2);
            double W2  = 1.0 / (1.0 - v2);
            double W   = std::sqrt(W2);
            double e   = (Tau + D * (1.0 - W) + p * (1.0 - W2)) / (D * W);
            double Rho = D / W;
            double h   = 1.0 + e + p / Rho;
            double cs2 = gm * p / (Rho * h);

            f = Rho * e * (gm - 1.0) - p;
            g = v2 * cs2 - 1.0;
            p -= f / g;

            if (std::fabs(f) < errorTolerance)
            {
                W_soln = W;
                soln_found = 1;
            }
            if (++n_iter == newtonIterMax)
            {
                throw std::invalid_argument("c2p failure: root finder not converging U=" + to_string(U));
            }
        }

        auto P = Vars();

        P[RHO] = D / W_soln;
        P[PRE] = p;
        P[V11] = U[S11] / (Tau + D + p);
        P[V22] = U[S22] / (Tau + D + p);
        P[V33] = U[S33] / (Tau + D + p);

        if (P[PRE] < 0.0)
        {
            throw std::invalid_argument("c2p failure: negative pressure U=" + to_string(U));
        }
        if (P[RHO] < 0.0)
        {
            throw std::invalid_argument("c2p failure: negative density U=" + to_string(U));
        }
        if (W_soln != W_soln || W_soln > maxW)
        {
            throw std::invalid_argument("c2p failure: nan W U=" + to_string(U));
        }
        return P;
    }

    const int newtonIterMax = 50;
    const double errorTolerance = 1e-12;
    const double maxW = 1e12;
    const double gammaLawIndex = 4. / 3;
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
        const double cs2 = gm0 * P[PRE] / P[RHO];
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
        const double dg = P[0];
        const double vr = P[1];
        const double vq = P[2];
        const double vp = P[3];
        const double pg = P[4];
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
    std::snprintf(res, 1024, "[%4.3lf %4.3lf %4.3lf %4.3lf %4.3lf]",
        V[0], V[1], V[2], V[3], V[4]);
    return res;
}
