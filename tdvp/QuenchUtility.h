#ifndef __QUENCHUTILITY_H_CMC__
#define __QUENCHUTILITY_H_CMC__
#include "itensor/all.h"
#include "ReadInput.h"
#include "IUtility.h"
#include "TDVPWorker.h"
#include "MyLocalmpo.h"
#include "GlobalIndices.h"
#include "GenMPO.h"
#include "FixedPointTensor.h"
#include "uGauge.h"
#include "iTDVP.h"
#include "MPSUtility.h"
#include "Entanglement.h"
#include "GeneralUtility.h"
using namespace itensor;
using namespace std;

vector<Index> get_site_inds (const MPS& psi)
{
    int N = length(psi);
    vector<Index> sites (N);
    for(int i = 1; i <= N; i++)
    {
        auto ii = findIndex (psi(i), "Site");
        sites.at(i-1) = ii;
    }
    return sites;
}

Fermion get_SiteSet (const MPS& mps)
{
    auto inds = get_site_inds (mps);
    return Fermion (inds);
}

ITensor get_W (const MPO& H, int i, const GlobalIndices& IS)
{
    auto W = H(i);
    auto iWl = leftLinkIndex (H, i);
    auto iWr = rightLinkIndex (H, i);
    auto is = findIndex (W, "Site,0");

    auto is2 = IS.is();
    auto iwl2 = IS.iwl();
    auto iwr2 = IS.iwr();
    auto is2pr = prime(is2);
    W.replaceInds ({is, prime(is), iWl, iWr}, {is2, is2pr, iwl2, iwr2});
    return W;
}

inline ITensor to_itdvp_AR (const ITensor& AR, const Index& iAl, const Index& iAr, const GlobalIndices& IS)
{
    auto ir = IS.ir();
    auto is = IS.is();
    auto iAs = findIndex (AR, "Site");
    auto ARre = replaceInds (AR, {iAl, iAr, iAs}, {prime(ir,2), ir, is});
    IS.check ("AR",ARre);
    return ARre;
}

inline ITensor to_itdvp_AL (const ITensor& AL, const Index& iAl, const Index& iAr, const GlobalIndices& IS)
{
    auto il = IS.il();
    auto is = IS.is();
    auto iAs = findIndex (AL, "Site");
    auto ALre = replaceInds (AL, {iAl, iAr, iAs}, {il, prime(il,2), is});
    IS.check ("AL",ALre);
    return ALre;
}

ITensor get_L (const ITensor& AR, const Index& iAl, const Index& iAr, const GlobalIndices& IS, Real crit=1e-15)
{
    auto il = IS.il();
    auto ir = IS.ir();
    auto is = IS.is();
    auto iAs = findIndex (AR, "Site");
    auto ARt = replaceInds (AR, {iAl, iAr, iAs}, {il, prime(il,2), is});

    auto [AL, C] = Orthogonalize (ARt, IS, crit);
    C.replaceInds ({il}, {ir});

    auto Cdag = dag(C);
    Cdag.prime (ir);

    auto L = C * Cdag;
    IS.check ("L",L);
    assert (check_leading_eigen (AR, AR, L));
    return L;
}

ITensor get_R (const ITensor& AL, const Index& iAl, const Index& iAr, const GlobalIndices& IS, Real crit=1e-15)
{
    auto il = IS.il();
    auto is = IS.is();
    auto iAs = findIndex (AL, "Site");
    auto ALt = replaceInds (AL, {iAl, iAr, iAs}, {prime(il,2), il, is});

    auto [AR, C] = Orthogonalize (ALt, IS, crit);
    C.mapPrime(1,2);

    auto Cdag = dag(C);
    Cdag.mapPrime(0,2);

    auto R = C * Cdag;
    IS.check ("R",R);
    assert (check_leading_eigen (AL, AL, R));
    return R;
}

void
expandL
(MPS& psi, MPO& H, unique_ptr<MyLocalMPO>& PH, int n, int NumCenter,
 const ITensor& AL, const Index& iALl, const Index& iALr)
{
    int oc = orthoCenter (psi);
    PH->position (oc, psi);

    int N = length (psi);
    int N2 = N + n;

    auto WL = H(1);

    auto iALs = findIndex (AL, "Site");
    auto iWLl = iut::leftIndex  (H, 1);
    auto iWLr = iut::rightIndex (H, 1);
    auto iWLs = findIndex (H(1), "Site,0");

    // Set MPS and MPO
    MPS psi2 (N2);
    MPO H2 (N2);
    // Insert original tensors
    for(int i = 1; i <= N; i++)
    {
        psi2.ref(n+i) = psi(i);
        H2.ref(n+i) = H(i);
    }
    // Insert new tensor to the left
    auto il0 = iut::leftIndex (psi, 1);
    auto iHl0 = iut::leftIndex (H, 1);
    auto il = sim (il0);
    auto iHl = sim (iHl0);
    psi2.ref(n+1).replaceInds ({il0}, {il});
    H2.ref(n+1).replaceInds ({iHl0}, {iHl});
    for(int i = n; i >= 1; i--)
    {
        auto is2 = sim(iALs);
        // Replace the site and the right indices
        psi2.ref(i) = replaceInds (AL, {iALs, iALr}, {is2, dag(il)});
        H2.ref(i)   = replaceInds (WL, {iWLs, prime(iWLs), iWLr}, {is2, prime(is2), dag(iHl)});
        il  = sim (iALl);
        iHl = sim (iWLl);
        if (i != 1)
        {
            psi2.ref(i).replaceInds ({iALl}, {il});
            H2.ref(i).replaceInds ({iWLl}, {iHl});
        }
    }
    psi2.rightLim (n+2);
    psi2.position (n+1);
    psi = psi2;
    H = H2;

    // Set PH
    Args args = {"NumCenter",NumCenter};
    auto PH2 = make_unique <MyLocalMPO> (H, PH->L(1), PH->R(N), args);
    for(int i = N-1; i >= 1; i--)
        PH2->setR (i+n, PH->R(i));
    PH2->setRHlim (n+1+NumCenter);
    PH2->position (n+1, psi);
    PH = move (PH2);
}

void
expandR
(MPS& psi, MPO& H, unique_ptr<MyLocalMPO>& PH, int n, int NumCenter,
 const ITensor& AR, const Index& iARl, const Index& iARr)
{
    int oc = orthoCenter (psi);
    PH->position (oc, psi);

    int N = length (psi);
    int N2 = N + n;

    auto WR = H(N);

    auto iARs = findIndex (AR, "Site");
    auto iWRl = iut::leftIndex  (H, N);
    auto iWRr = iut::rightIndex (H, N);
    auto iWRs = findIndex (H(N), "Site,0");

    // Set MPS and MPO
    MPS psi2 (N2);
    MPO H2 (N2);
    // Insert original tensors
    for(int i = 1; i <= N; i++)
    {
        psi2.ref(i) = psi(i);
        H2.ref(i) = H(i);
    }
    // Insert new tensors to the right
    auto ir0  = iut::rightIndex (psi, N);
    auto iHr0 = iut::rightIndex (H, N);
    auto ir = sim (ir0);
    auto iHr = sim (iHr0);
    psi2.ref(N).replaceInds ({ir0}, {ir});
    H2.ref(N).replaceInds ({iHr0}, {iHr});
    for(int i = N+1; i <= N2; i++)
    {
        auto is2 = sim(iARs);
        // Replace the site and the left indices
        psi2.ref(i) = replaceInds (AR, {iARs, iARl}, {is2, dag(ir)});
        H2.ref(i)   = replaceInds (WR, {iWRs, prime(iWRs), iWRl}, {is2, prime(is2), dag(iHr)});
        ir  = sim (iARr);
        iHr = sim (iWRr);
        if (i != N2)
        {
            psi2.ref(i).replaceInds ({iARr}, {ir});
            H2.ref(i).replaceInds ({iWRr}, {iHr});
        }
    }
    psi2.leftLim (N-1);
    psi2.position(N);
    psi = psi2;
    H = H2;

    // Set PH
    Args args = {"NumCenter",NumCenter};
    auto PH2 = make_unique <MyLocalMPO> (H, PH->L(1), PH->R(N), args);
    for(int i = 2; i <= N; i++)
        PH2->setL (i, PH->L(i));
    PH2->setLHlim (N-NumCenter);
    PH2->position (N, psi2);
    PH = move (PH2);
}

void get_init (const string& infile,
               MPS& psi, MPO& H, Fermion& sites,
               Index& il, Index& ir,
               ITensor& WL, ITensor& WR,
               ITensor& AL_left,  ITensor& AR_left,  ITensor& AC_left,  ITensor& C_left,
               ITensor& La_left,  ITensor& Ra_left,  ITensor& LW_left,  ITensor& RW_left,
               ITensor& AL_right, ITensor& AR_right, ITensor& AC_right, ITensor& C_right,
               ITensor& La_right, ITensor& Ra_right, ITensor& LW_right, ITensor& RW_right,
               GlobalIndices& IS,
               int& idev_first, int& idev_last)
{
    InputGroup input (infile,"basic");

    auto L_lead   = input.getInt("L_lead");
    auto L_device   = input.getInt("L_device");
    auto t_lead     = input.getReal("t_lead");
    auto t_device   = input.getReal("t_device");
    auto t_contactL  = input.getReal("t_contactL");
    auto t_contactR  = input.getReal("t_contactR");
    auto mu_leadL   = input.getReal("mu_leadL");
    auto mu_leadR   = input.getReal("mu_leadR");
    auto mu_device  = input.getReal("mu_device");
    auto V_lead     = input.getReal("V_lead");
    auto V_device   = input.getReal("V_device");
    auto V_contact  = input.getReal("V_contact");

    auto psi_dir        = input.getString("psi_dir");
    auto psi_file       = input.getString("psi_file");
    auto itdvp_dir      = input.getString("itdvp_dir");
    auto ErrGoal        = input.getReal("ErrGoal");
    auto MaxIter_LRW    = input.getInt("MaxIter_LRW");

    // Read the iTDVP tensors
    ITensor AL, AR, AC, C, LW, RW, La, Ra;
    readFromFile (itdvp_dir+"/AL.itensor", AL);
    readFromFile (itdvp_dir+"/AR.itensor", AR);
    readFromFile (itdvp_dir+"/AC.itensor", AC);
    readFromFile (itdvp_dir+"/C.itensor", C);
    readFromFile (itdvp_dir+"/LW.itensor", LW);
    readFromFile (itdvp_dir+"/RW.itensor", RW);
    readFromFile (itdvp_dir+"/La.itensor", La);
    readFromFile (itdvp_dir+"/Ra.itensor", Ra);
    IS.read (itdvp_dir+"/global.inds");
    il = IS.il();
    ir = IS.ir();
    int dim = ir.dim();
    // Read MPS in the window
    readFromFile (psi_dir+"/"+psi_file, psi);
    int N = length (psi);
    // Site indices
    auto site_inds = get_site_inds (psi);
    sites = Fermion (site_inds);
    // Hamiltonian
    AutoMPO ampo;
    tie (ampo, idev_first, idev_last)
    = t_mu_V_ampo (sites, L_lead, L_device,
                   t_lead, t_lead, t_device, t_contactL, t_contactR,
                   mu_leadL, mu_leadR, mu_device,
                   V_lead, V_lead, V_device, V_contact, V_contact);
    auto locamu = read_bracket_values<int,Real> (infile, "local_mu", 1);
    auto localtV = read_bracket_values<int,int,Real,Real> (infile, "local_tV", 1);
    ampo_add_mu (ampo, locamu, true);
    ampo_add_tV (ampo, localtV, true);
    H = toMPO (ampo);
    to_inf_mpo (H, IS.iwl(), IS.iwr());
    // Boundary tensors
    cout << "Compute boundary tensors" << endl;
    WL = get_W (H, 2, IS);
    WR = get_W (H, N-1, IS);
    Args args_itdvp = {"ErrGoal=",ErrGoal,"MaxIter",MaxIter_LRW};
    auto L = get_LR <LEFT> (C, IS);
    auto R = get_LR <RIGHT> (C, IS);
    Real enL, enR;
    tie (LW, enL) = get_LRW <LEFT>  (AL, WL, R, La, IS, args_itdvp);
    tie (RW, enR) = get_LRW <RIGHT> (AR, WR, L, Ra, IS, args_itdvp);
    // Check
    mycheck (commonIndex (LW, H(1)), "LW and H(1) has no common Index");
    mycheck (commonIndex (RW, H(N)), "RW and H(N) has no common Index");
    mycheck (commonIndex (LW, psi(1)), "LW and psi(1) has no common Index");
    mycheck (commonIndex (RW, psi(N)), "RW and psi(N) has no common Index");
    // Left boundaries
    AL_left = AL,
    AR_left = AR,
    AC_left = AC,
    C_left = C,
    La_left = La,
    Ra_left = Ra,
    LW_left = LW,
    RW_left = RW;
    // Right boundaries
    AL_right = AL,
    AR_right = AR,
    AC_right = AC,
    C_right = C,
    La_right = La,
    Ra_right = Ra,
    LW_right = LW,
    RW_right = RW;
}

template<typename T>
void writeAll (ostream& os, const T& t)
{
    write (os, t);
}

template<typename T, typename... Args>
void writeAll (ostream& os, const T& first, const Args&... args)
{
    write (os, first);
    writeAll (os, args...);
}

template<typename T>
void readAll (istream& is, T& t)
{
    read (is, t);
}

template<typename T, typename... Args>
void readAll (istream& is, T& first, Args&... args)
{
    read (is, first);
    readAll (is, args...);
}
#endif
