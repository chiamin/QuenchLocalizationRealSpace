#include <typeinfo>
#include <iomanip>
#include "itensor/all.h"
#include "IUtility.h"
#include "itdvp/uGauge.h"
#include "itdvp/GenMPO.h"
#include "itdvp/FixedPointTensor.h"
#include "itdvp/Solver.h"
#include "itdvp/iTDVP.h"
#include "itdvp/GlobalIndices.h"
#include "dmrg/MyObserver.h"
#include "dmrg/ToMPS.h"
#include "tdvp/QuenchUtility.h"
#include "tdvp/TDVPObserver.h"
using namespace itensor;
using namespace std;


tuple<ITensor,ITensor,ITensor,ITensor,ITensor,ITensor,ITensor,ITensor,ITensor,GlobalIndices>
itdvp (Real t, Real mu, Real V, string infile)
{
    InputGroup input (infile,"itdvp");

    auto time_steps  = input.getInt("time_steps");
    auto dt_str      = input.getString("dt");
    auto dt = ((dt_str == "inf" || dt_str == "Inf" || dt_str == "INF") ? INFINITY : stod(dt_str));
    auto D           = input.getInt("max_dim");
    auto ErrGoal     = input.getReal("ErrGoal");
    auto MaxIter     = input.getInt("MaxIter");
    auto ErrGoalInit = input.getReal("ErrGoalInit",1e-12);
    auto MaxIterInit = input.getInt("MaxIterInit",100);
    auto SeedInit    = input.getInt("SeedInit",0);

    //------------------------------------------
    cout << setprecision(18) << endl;
    // Make MPO
    auto sites = Fermion (3, {"ConserveQNs",false});

    auto H = single_empurity_mpo (sites, 2, t, t, t, mu, mu, mu, V, V, V);

    auto [W, is, iwl, iwr] = get_W (H);
    // W: MPO tensor
    // is: physical index
    // iwl: left index
    // iwr: right index

    // Initialize MPS
    auto A = ITensor(); // ill-defined tensor
    auto [AL, AR, AC, C, La0, Ra0, IS] = itdvp_initial (W, is, iwl, iwr, A, D, ErrGoalInit, MaxIterInit, SeedInit);
    // If A is ill-defined, A will be random generated in itdvp_initial
    // This is just for tensors to be used in iTDVP

    // iTDVP
    Args args = {"ErrGoal=",1e-4,"MaxIter",MaxIter};
    ITensor LW, RW;
    Real en, err;
    for(int i = 1; i <= time_steps; i++)
    {
        cout << "time step " << i << endl;
        // Run iTDVP
        // If dt is real,       do imaginary time evolution
        //          imaginary,     real
        tie (en, err, LW, RW) = itdvp (W, AL, AR, AC, C, La0, Ra0, dt, IS, args);
        cout << "energy, error = " << en << " " << err << endl;

        // Decrease the ErrGoal dynamically
        if (args.getReal("ErrGoal") > ErrGoal)
            args.add("ErrGoal=",err*0.1);
    }
    return {AL, AR, AC, C, LW, RW, W, La0, Ra0, IS};
}

MPS dmrg_inf (string infile, const vector<Real>& mus_device,
              ITensor AL_L, ITensor AR_L, ITensor AC_L, ITensor LW_L, ITensor RW_L,
              ITensor AL_R, ITensor AR_R, ITensor AC_R, ITensor LW_R, ITensor RW_R,
              const GlobalIndices& ISL, const GlobalIndices& ISR)
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

    InputGroup input2 (infile,"dmrg");
    auto WriteDim   = input2.getInt("WriteDim",-1);
    auto ConserveQNs = input2.getYesNo("ConserveQNs",false);
    auto sweeps     = iut::Read_sweeps (infile, "sweeps_dmrg");

    // Site set
    using SitesType = Fermion;
    auto sites = SitesType (2*L_lead+L_device, {"ConserveQNs",ConserveQNs});

    // Initialze MPS
    MPS psi = toMPS (sites, AL_L, ISL.il(),          prime(ISL.il(),2), ISL.is(),
                            AR_R, prime(ISR.ir(),2), ISR.ir(),          ISR.is(),
                            AC_L, ISL.il(),          ISL.ir(),          ISL.is());
    psi.position(1);
    assert (commonIndex (LW_L, psi(1)));
    assert (commonIndex (RW_R, psi(length(psi))));

    // Make MPO
    auto [ampo, idev_first, idev_last]
    = t_mu_V_ampo (sites, L_lead, L_device,
                   t_lead, t_lead, t_device, t_contactL, t_contactR,
                   mu_leadL, mu_leadR, mus_device,
                   V_lead, V_lead, V_device, V_contact, V_contact);
    auto H = toMPO (ampo);
    to_inf_mpo (H, ISL.iwl(), ISR.iwr());
    assert (commonIndex (LW_L, H(1)));
    assert (commonIndex (RW_R, H(length(H))));

    // DMRG
    MyObserver<SitesType> myobs (sites, psi, {"Write",false});
    dmrg (psi, H, LW_L, RW_R, sweeps, myobs, {"WriteDim",WriteDim});

    return psi;
}

void get_init_quench
(const string& infile, const Fermion& sites, const vector<Real>& mus_device,
 ITensor AL, ITensor AR, ITensor AC, ITensor C, ITensor LW, ITensor RW, ITensor La, ITensor Ra, const GlobalIndices& IS,
 MPO& H, Index& il, Index& ir,
 ITensor& WL, ITensor& WR,
 ITensor& AL_left,  ITensor& AR_left,  ITensor& AC_left,  ITensor& C_left,
 ITensor& La_left,  ITensor& Ra_left,  ITensor& LW_left,  ITensor& RW_left,
 ITensor& AL_right, ITensor& AR_right, ITensor& AC_right, ITensor& C_right,
 ITensor& La_right, ITensor& Ra_right, ITensor& LW_right, ITensor& RW_right,
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
    auto mu_biasL   = input.getReal("mu_biasL");
    auto mu_biasS   = input.getReal("mu_biasS");
    auto mu_biasR   = input.getReal("mu_biasR");

    InputGroup input2 (infile,"tdvp");
    auto ErrGoal_LRW    = input2.getReal("ErrGoal_LRW");
    auto MaxIter_LRW    = input2.getInt("MaxIter_LRW");

    il = IS.il();
    ir = IS.ir();

    int dim = ir.dim();
    int N = length (sites);

    // device mu with bias
    auto mu_device_bias = mus_device;
    for(auto& mu : mu_device_bias)
        mu += mu_biasS;

    // Hamiltonian
    AutoMPO ampo;
    tie (ampo, idev_first, idev_last)
    = t_mu_V_ampo (sites, L_lead, L_device,
                   t_lead, t_lead, t_device, t_contactL, t_contactR,
                   mu_leadL+mu_biasL, mu_leadR+mu_biasR, mu_device_bias,
                   V_lead, V_lead, V_device, V_contact, V_contact);
    auto locamu = read_bracket_values<int,Real> (infile, "local_mu", 1);
    auto localtV = read_bracket_values<int,int,Real,Real> (infile, "local_tV", 1);
    ampo_add_mu (ampo, locamu, true);
    ampo_add_tV (ampo, localtV, true);
    H = toMPO (ampo);
    to_inf_mpo (H, IS.iwl(), IS.iwr());

    // Boundary tensors
    cout << "Compute boundary tensors" << endl;
    WL = get_W (H, 2, IS);      // from left lead  
    WR = get_W (H, N-1, IS);    // from right lead
    Args args_itdvp = {"ErrGoal=",ErrGoal_LRW,"MaxIter",MaxIter_LRW};
    auto L = get_LR <LEFT> (C, IS);
    auto R = get_LR <RIGHT> (C, IS);
    Real enL, enR;
    tie (LW, enL) = get_LRW <LEFT>  (AL, WL, R, La, IS, args_itdvp);
    tie (RW, enR) = get_LRW <RIGHT> (AR, WR, L, Ra, IS, args_itdvp);
    // Check
    mycheck (commonIndex (LW, H(1)), "LW and H(1) has no common Index");
    mycheck (commonIndex (RW, H(N)), "RW and H(N) has no common Index");
    //mycheck (commonIndex (LW, psi(1)), "LW and psi(1) has no common Index");
    //mycheck (commonIndex (RW, psi(N)), "RW and psi(N) has no common Index");
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

void tdvp_quench (string infile, MPS psi, vector<Real>& mus_device,
                  ITensor AL, ITensor AR, ITensor AC, ITensor C, ITensor LW, ITensor RW, ITensor La, ITensor Ra, GlobalIndices IS)
{
    InputGroup input (infile,"tdvp");

    auto dt            = input.getReal("dt");
    auto time_steps    = input.getInt("time_steps");
    auto NumCenter     = input.getInt("NumCenter");
    auto ConserveQNs   = input.getYesNo("ConserveQNs",false);
    auto expandN       = input.getInt("expandN",0);
    auto expand_step   = input.getInt("expand_step",1);
    auto max_window    = input.getInt("max_window",10000);
    auto sweeps        = iut::Read_sweeps (infile, "sweeps_tdvp");

    auto ErrGoal_LRW   = input.getReal("ErrGoal_LRW");
    auto MaxIter_LRW   = input.getInt("MaxIter_LRW");
    auto UseSVD        = input.getYesNo("UseSVD",true);
    auto SVDmethod     = input.getString("SVDMethod","gesdd");  // can be also "ITensor"
    auto WriteDim      = input.getInt("WriteDim");

    InputGroup input2 (infile,"basic");

    auto write         = input2.getYesNo("write",false);
    auto write_dir     = input2.getString("write_dir",".");
    auto write_file    = input2.getString("write_file","");
    auto read          = input2.getYesNo("read",false);
    auto read_dir      = input2.getString("read_dir",".");
    auto read_file     = input2.getString("read_file","");

    auto out_dir     = input.getString("outdir",".");
    if (write_dir == "." && out_dir != ".")
        write_dir = out_dir;

    // Declare variables
    MPO H;
    Fermion sites (get_site_inds (psi));
    Index   il, ir;
    ITensor WL, WR,
            AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
            AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right;
    bool expand = false,
         expand_next = false;
    int step = 1;
    int idev_first, idev_last;

    // Initialize variables
    if (!read)
    {
        mycheck (commonIndex (LW, psi(1)), "LW and psi(1) has no common Index");
        mycheck (commonIndex (RW, psi(length(psi))), "RW and psi(N) has no common Index");
        get_init_quench (infile, sites, mus_device,
                         AL, AR, AC, C, LW, RW, La, Ra, IS,
                         H, il, ir, WL, WR,
                         AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                         AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right,
                         idev_first, idev_last);
    }
    // Read variables
    else
    {
        ifstream ifs = open_file (read_dir+"/"+read_file);
        readAll (ifs, psi, H, il, ir, WL, WR,
                 AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                 AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right,
                 step, idev_first, idev_last, expand, expand_next, IS, mus_device);
        auto site_inds = get_site_inds (psi);
        sites = Fermion (site_inds);
    }

    // Args parameters
    Args args_itdvp = {"ErrGoal=",ErrGoal_LRW,"MaxIter",MaxIter_LRW};
    Args args_obs   = {"ConserveQNs",ConserveQNs};
    Args args_tdvp  = {"Quiet",true,"NumCenter",NumCenter,"DoNormalize",true,"UseSVD",UseSVD,"SVDmethod",SVDmethod,"WriteDim",WriteDim};

    // Observer
    auto obs = make_unique <TDVPObserver> (sites, psi, args_obs);

    // Effective Hamiltonian
    auto PH = make_unique <MyLocalMPO> (H, LW_left, RW_right, args_tdvp);

    // Time evolution
    cout << "Start time evolution" << endl;
    cout << sweeps << endl;
    psi.position(1);
    Real en, err;
    int N = length (psi);

    for(int i = 0; i < time_steps; i++)
    {
        cout << "step = " << step++ << endl;

        auto expand = (expandN != 0 && step % expand_step == 0 && N < max_window);
        // Extend left edge
        if (expand)
        {
            // Expand
            expandL (psi, H, PH, expandN, NumCenter, AL_left, il, prime(il,2));
            idev_first += expandN;
            idev_last += expandN;
            cout << "expand left " << expandN << endl;
            // Observer
            sites = get_SiteSet (psi);
            obs = make_unique <TDVPObserver> (sites, psi, args_obs);
            N = length (psi);
            psi.position(1);
        }
        cout << "device site = " << idev_first << " " << idev_last << endl;

        // Evolve left edge
        tie (en, err, LW_left, RW_left) = itdvp (WL, AL_left, AR_left, AC_left, C_left, La_left, Ra_left, 1_i*dt, IS, args_itdvp);
        PH->L (1, LW_left);

        // From left to right
        TDVPWorker <Fromleft>  (psi, *PH, 1_i*dt, sweeps, *obs, args_tdvp);


        // Extend right boundary
        if (expand)
        {
            // Expand
            expandR (psi, H, PH, expandN, NumCenter, AR_right, prime(ir,2), ir);
            // Observer
            sites = get_SiteSet (psi);
            obs = make_unique <TDVPObserver> (sites, psi, args_obs);
            N = length (psi);
            psi.position (N);
        }

        // Evolve right boundary
        tie (en, err, LW_right, RW_right) = itdvp (WR, AL_right, AR_right, AC_right, C_right, La_right, Ra_right, 1_i*dt, IS, args_itdvp);
        PH->R (N, RW_right);

        // From right to left
        TDVPWorker <Fromright> (psi, *PH, 1_i*dt, sweeps, *obs, args_tdvp);

        if (write)
        {
            ofstream ofs (write_dir+"/"+write_file);
            writeAll (ofs, psi, H, il, ir, WL, WR,
                      AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                      AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right,
                      step, idev_first, idev_last, expand, expand_next, IS, mus_device);
        }
    }
}

int main(int argc, char* argv[])
{
    string infile = argv[1];
    // Decide to read or not
    InputGroup input (infile,"basic");
    auto read = input.getYesNo("read",false);

    ITensor AL, AR, AC, C, LW, RW, W, La0, Ra0;
    GlobalIndices IS;
    MPS psi;
    vector<Real> mus_device;
    if (not read)
    {
        // Find the ground state of the infinite-length leads by iTDVP
        auto t_lead  = input.getReal("t_lead");
        auto mu_lead = input.getReal("mu_leadL");
        auto V_lead  = input.getReal("V_lead");
        tie (AL, AR, AC, C, LW, RW, W, La0, Ra0, IS) = itdvp (t_lead, mu_lead, V_lead, infile);

        // Set the random chemical potential
        auto L_device   = input.getInt("L_device");
        auto mu_device  = input.getReal("mu_device");
        auto W_device   = input.getReal("disorder_strength");
        auto seed       = input.getInt("seed",time(NULL));
        std::mt19937 rgen(seed);
        std::uniform_real_distribution<> dist (-W_device,W_device); // distribution in range [-W_device, W_device]
        cout << "Disordered chemical potential" << endl;
        for(int i = 0; i < L_device; i++)
        {
            auto mui = mu_device + dist (rgen);
            mus_device.push_back (mui);
            cout << i+1 << " " << mui << endl;
        }

        // Find the ground state of the whole system by DMRG
        psi = dmrg_inf (infile, mus_device,
                           AL, AR, AC, LW, RW,
                           AL, AR, AC, LW, RW,
                           IS, IS);
    }

    // Do quench dynamics using TDVP
    tdvp_quench (infile, psi, mus_device, AL, AR, AC, C, LW, RW, La0, Ra0, IS);

    return 0;
}
