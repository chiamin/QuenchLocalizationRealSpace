#ifndef __ITDVP_H_CMC__
#define __ITDVP_H_CMC__
#include <typeinfo>
#include <iomanip>
#include "itensor/all.h"
#include "Solver.h"
#include "RandomUtility.h"
using namespace itensor;
using namespace std;

tuple <ITensor, ITensor, ITensor, ITensor, ITensor, ITensor, GlobalIndices>
itdvp_initial (const ITensor& W, const Index& is, const Index& iwl, const Index& iwr, ITensor& A, int D, Real ErrGoal, int MaxIter,
               RandGen::SeedType seed=0)
{
    Index il;
    if (!A)
    {
        il = Index(D,"Link");
        A = ITensor (is, il, prime(il,2));
        auto rand = RandGen (seed);
        auto gen = [&rand]() { return rand.real(); };
        A.generate (gen);
        //A.randomize();
        //A.set(1,1,1,1.);
    }
    else
    {
        il = findIndex (A, "Link,0");
        if (!hasIndex (A, prime(il,2)))
        {
            cout << "Error: " << __FUNCTION__ << ": A has wrong index structure" << endl;
            throw;
        }
    }
    auto ir = sim(il);

    GlobalIndices IS;
    IS._iwl = iwl;
    IS._iwr = iwr;
    IS._is = is;
    IS._il = il;
    IS._ir = ir;

    auto [AL, AR, C] = MixedCanonical (A, IS, ErrGoal, MaxIter);

    auto AC = get_AC (AL, C, IS);

    auto La0 = randomITensor (il, prime(il));
    auto Ra0 = randomITensor (ir, prime(ir));

    return {AL, AR, AC, C, La0, Ra0, IS};
}

inline Real diff_ALC_AC (const ITensor& AL, const ITensor& C, const ITensor& AC, const GlobalIndices& IS)
{
    auto ALC = get_AC (AL, C, IS);
    auto d = norm (ALC - AC);
    return d;
}

template <typename TimeType>
tuple <Real, Real, ITensor, ITensor>
itdvp
(const ITensor& W, ITensor& AL, ITensor& AR, ITensor& AC, ITensor& C, ITensor& La0, ITensor& Ra0, TimeType dt, const GlobalIndices& IS,
 Args& args=Args::global())
// args: ErrGoal, MaxIter, used in applyExp and arnoldi
{
    // C --> L, R
    auto L = get_LR <LEFT>  (C, IS);
    auto R = get_LR <RIGHT> (C, IS);

    // AL, AR, L, R --> LW, RW, enL, enR
    auto [LW, enL] = get_LRW <LEFT>  (AL, W, R, La0, IS, args);
    auto [RW, enR] = get_LRW <RIGHT> (AR, W, L, Ra0, IS, args);

    auto en = 0.5*(enL + enR);

    // LW, RW, W, AC, C --> AC, C
    if constexpr(is_same_v<TimeType,Real>)
    {
        if (isinf (dt))
            solve_gs (LW, RW, W, AC, C, IS, args);
        else
            time_evolve (LW, RW, W, AC, C, dt, IS, args);
    }
    else
    {
        time_evolve (LW, RW, W, AC, C, dt, IS, args);
    }
    // AC, C --> AL, AR
    AL = get_AL (AC, C, IS);
    AR = get_AR (AC, C, IS);
    //AL = get_ALR_2 <LEFT>  (AC, C, IS);
    //AR = get_ALR_2 <RIGHT> (AC, C, IS);

    C /= norm(C);
    AC /= norm(AC);

    auto errL = diff_ALC_AC (AL, C, AC, IS);
    auto errR = diff_ALC_AC (AR, C, AC, IS);
    auto err = (errL > errR ? errL : errR);

    return {en, err, LW, RW};
}
#endif
