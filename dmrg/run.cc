#include "itensor/all.h"
#include "ReadInput.h"
#include "IUtility.h"
#include "MyObserver.h"
#include "../itdvp/GlobalIndices.h"
#include "GenMPO.h"
#include "ToMPS.h"
using namespace itensor;
using namespace std;

template <typename SitesType>
MPO current_correlation (const SitesType& sites, int i)
{
    int N = length(sites);
    AutoMPO ampo (sites);
    if constexpr (is_same_v <SitesType, Fermion>)
    {
        ampo += -1_i,"Cdag",i,"C",i+1;
        ampo +=  1_i,"Cdag",i+1,"C",i;
        ampo +=  1_i,"Cdag",N-i,"C",N-i+1;
        ampo += -1_i,"Cdag",N-i+1,"C",N-i;
    }

    return toMPO (ampo);
}

Cplx myinner (const MPO& mpo, const MPS& psi)
{
    auto L = ITensor(1.);
    int N = length(psi);
    for(int i = 1; i <= N; i++)
    {
        L *= psi(i) * mpo(i);
        auto dagA = dag(psi(i));
        dagA.prime ("Site");
        if (i == 1)
        {
            auto ii = commonIndex (psi(i), psi(i+1), "Link");
            dagA.prime (ii);
        }
        else if (i == N)
        {
            auto ii = commonIndex (psi(i), psi(i-1), "Link");
            dagA.prime (ii);
        }
        else
        {
            dagA.prime ("Link");
        }
        L *= dagA;
    }
    return eltC(L);
}

int main(int argc, char* argv[])
{
    string infile = argv[1];
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

    auto read_dir_L = input.getString("read_dir_L");
    auto read_dir_R = input.getString("read_dir_R");

    auto WriteDim   = input.getInt("WriteDim",-1);
    auto do_write   = input.getYesNo("write_to_file");
    auto out_dir    = input.getString("outdir",".");
    auto out_minm   = input.getInt("out_minm",0);
    auto H_file     = input.getString("H_outfile","H.mpo");
    auto ConserveQNs = input.getYesNo("ConserveQNs",false);
    auto sweeps     = iut::Read_sweeps (infile);

    // Read the tensors
    ITensor AL_L, AR_L, AC_L, LW_L, RW_L;
    ITensor AL_R, AR_R, AC_R, LW_R, RW_R;
    readFromFile (read_dir_L+"/AL.itensor", AL_L);
    readFromFile (read_dir_L+"/AR.itensor", AR_L);
    readFromFile (read_dir_L+"/AC.itensor", AC_L);
    readFromFile (read_dir_L+"/LW.itensor", LW_L);
    readFromFile (read_dir_L+"/RW.itensor", RW_L);
    readFromFile (read_dir_R+"/AL.itensor", AL_R);
    readFromFile (read_dir_R+"/AR.itensor", AR_R);
    readFromFile (read_dir_R+"/AC.itensor", AC_R);
    readFromFile (read_dir_R+"/LW.itensor", LW_R);
    readFromFile (read_dir_R+"/RW.itensor", RW_R);
    GlobalIndices ISL, ISR;
    ISL.read (read_dir_L+"/global.inds");
    ISR.read (read_dir_R+"/global.inds");

    // Site set
    using SitesType = Fermion;
    auto sites = SitesType (2*L_lead+L_device, {"ConserveQNs",ConserveQNs});

    // Initialze MPS
    MPS psi = toMPS (sites, AL_L, ISL.il(),          prime(ISL.il(),2), ISL.is(),
                            AR_R, prime(ISR.ir(),2), ISR.ir(),          ISR.is(),
                            AC_L, ISL.il(),          ISL.ir(),          ISL.is());
    psi.position(1);
    assert (commonIndex (LW_L, psi(1)));
    assert (commonIndex (RW_R, psi(L)));

    // Make MPO
    auto [ampo, idev_first, idev_last]
    = t_mu_V_ampo (sites, L_lead, L_device,
                   t_lead, t_lead, t_device, t_contactL, t_contactR,
                   mu_leadL, mu_leadR, mu_device,
                   V_lead, V_lead, V_device, V_contact, V_contact);
    auto H = toMPO (ampo);
    to_inf_mpo (H, ISL.iwl(), ISR.iwr());
    assert (commonIndex (LW_L, H(1)));
    assert (commonIndex (RW_R, H(L)));

    // Write to files
    writeToFile (out_dir+"/"+H_file, H);

    // DMRG
    MyObserver<SitesType> myobs (sites, psi, {"Write",do_write,"out_dir",out_dir,"out_minm",out_minm});
    dmrg (psi, H, LW_L, RW_R, sweeps, myobs, {"WriteDim",WriteDim});
/*
    int N = length(sites);
    for(int i = 1; i <= N/2; i++)
    {
        auto J = current_correlation (sites, i);
        auto j = myinner (J, psi);
        cout << "<J,J> " << i << " = " << j << endl;
    }
*/
    return 0;
}
