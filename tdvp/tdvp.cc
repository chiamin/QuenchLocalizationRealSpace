#include "itensor/all.h"
#include "QuenchUtility.h"
#include "TDVPObserver.h"
using namespace itensor;
using namespace std;

int main(int argc, char* argv[])
{
    string infile = argv[1];
    InputGroup input (infile,"basic");

    auto dt            = input.getReal("dt");
    auto time_steps    = input.getInt("time_steps");
    auto NumCenter     = input.getInt("NumCenter");
    auto ConserveQNs   = input.getYesNo("ConserveQNs",false);
    auto expandN       = input.getInt("expandN",0);
    auto expand_checkN = input.getInt("expand_checkN",5);
    auto expandS_crit  = input.getReal("expandS_crit",1e10);
    auto max_window    = input.getInt("max_window",10000);
    auto sweeps        = iut::Read_sweeps (infile);

    auto ErrGoal       = input.getReal("ErrGoal");
    auto MaxIter_LRW   = input.getInt("MaxIter_LRW");
    auto UseSVD        = input.getYesNo("UseSVD",true);
    auto SVDmethod     = input.getString("SVDMethod","gesdd");  // can be also "ITensor"
    auto WriteDim      = input.getInt("WriteDim");

    auto write         = input.getYesNo("write",false);
    auto write_dir     = input.getString("write_dir",".");
    auto write_file    = input.getString("write_file","");
    auto read          = input.getYesNo("read",false);
    auto read_dir      = input.getString("read_dir",".");
    auto read_file     = input.getString("read_file","");

    auto out_dir     = input.getString("outdir",".");
    if (write_dir == "." && out_dir != ".")
        write_dir = out_dir;

    // Declare variables
    MPS psi;
    MPO H;
    Fermion sites;
    Index   il, ir;
    ITensor WL, WR,
            AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
            AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right;
    GlobalIndices IS;
    bool expand = false,
         expand_next = false;
    int step = 1;
    int idev_first, idev_last;

    // Initialize variables
    if (!read)
    {
        get_init (infile,
                  psi, H, sites, il, ir, WL, WR,
                  AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                  AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right, IS,
                  idev_first, idev_last);
    }
    // Read variables
    else
    {
        ifstream ifs = open_file (read_dir+"/"+read_file);
        readAll (ifs,
                 psi, H, il, ir, WL, WR,
                 AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                 AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right,
                 step, idev_first, idev_last, expand, expand_next, IS);
        auto site_inds = get_site_inds (psi);
        sites = Fermion (site_inds);
    }

    // Args parameters
    Args args_itdvp = {"ErrGoal=",ErrGoal,"MaxIter",MaxIter_LRW};
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

        if (expandN != 0)
        {
            auto const& specR = obs->spec (N - expand_checkN);
            auto SR_sys = EntangEntropy (specR);
            auto SR = EntangEntropy_singular (C_right);
            expand_next = (abs(SR_sys-SR) > expandS_crit);
        }

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

        if (expandN != 0)
        {
            auto const& specL = obs->spec (expand_checkN);
            auto SL_sys = EntangEntropy (specL);
            auto SL = EntangEntropy_singular (C_left);
            if (abs(SL_sys-SL) > expandS_crit)
                expand_next = true;
        }

        //if (maxLinkDim(psi) >= sweeps.maxdim(1))
            //args_tdvp.add ("NumCenter",1);
        expand = expand_next;
        if (N >= max_window)
        {
            expand = false;
            expandN = 0;
        }

        if (write)
        {
            ofstream ofs (write_dir+"/"+write_file);
            writeAll (ofs,
                      psi, H, il, ir, WL, WR,
                      AL_left,  AR_left,  AC_left,  C_left,  La_left,  Ra_left,  LW_left,  RW_left,
                      AL_right, AR_right, AC_right, C_right, La_right, Ra_right, LW_right, RW_right,
                      step, idev_first, idev_last, expand, expand_next, IS);
        }
    }

    return 0;
}
