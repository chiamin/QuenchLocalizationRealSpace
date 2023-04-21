#ifndef __TOMPS_H_CMC__
#define __TOMPS_H_CMC__
#include "itensor/all.h"
using namespace itensor;
using namespace std;

MPS toMPS (const SiteSet& sites,
           const ITensor& AL, const Index& iALl, const Index& iALr, const Index& iALs,
           const ITensor& AR, const Index& iARl, const Index& iARr, const Index& iARs,
           const ITensor& AC, const Index& iACl, const Index& iACr, const Index& iACs,
           int oc=1)
{
    MPS psi (sites);
    int N = length (psi);

    // Set tensors
    vector<Index> ils(N+1), irs(N+1), iss(N+1);
    for(int i = 1; i <= N; i++)
    {
        Index il, ir;
        if (i < oc)
        {
            psi.ref(i) = AL;
            ils.at(i) = iALl;
            irs.at(i) = iALr;
            iss.at(i) = iALs;
        }
        else if (i == oc)
        {
            psi.ref(i) = AC;
            ils.at(i) = iACl;
            irs.at(i) = iACr;
            iss.at(i) = iACs;
        }
        else
        {
            psi.ref(i) = AR;
            ils.at(i) = iARl;
            irs.at(i) = iARr;
            iss.at(i) = iARs;
        }
        assert (hasIndex (psi(i), ils.at(i)));
        assert (hasIndex (psi(i), irs.at(i)));
    }

    // Replace indices
    auto inew = noPrime (sim (irs.at(1)));
    psi.ref(1).replaceInds ({iss.at(1), irs.at(1)}, {sites(1), inew});
    irs.at(1) = inew;
    for(int i = 2; i < N; i++)
    {
        inew = noPrime(sim(irs.at(i)));
        psi.ref(i).replaceInds ({iss.at(i), ils.at(i), irs.at(i)}, {sites(i), irs.at(i-1), inew});
        ils.at(i) = irs.at(i-1);
        irs.at(i) = inew;
    }
    psi.ref(N).replaceInds ({iss.at(N), ils.at(N)}, {sites(N), irs.at(N-1)});
    ils.at(N) = irs.at(N-1);

    // Check
    for(int i = 1; i <= N; i++)
    {
        assert (hasIndex (psi(i), sites(i)));
        if (i != N)
            assert (commonIndex (psi(i), psi(i+1)));
    }

    return psi;
}
#endif
