#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00

# Loop through the array and generate files with different kbT values
for ((i=0; i<=15; i++)); do
    omega=$(awk "BEGIN {print $i * 0.01}") # Corrected calculation of omega value
    for kbT in 0.015; do
        cat <<EOF > "primitive_Lindhard_Y2Ir2O7_omega=${omega}_kbT=${kbT}_n=8.py"

"""
Created on March 18th 10:02:16 2024
@author: amnedic
"""

import numpy as np
import scipy.linalg as la
import time
from multiprocessing import Pool
import numba as nb
from numba import njit
import gc

omega = $omega #finite frequency in eV
kbT = $kbT #temperature in eV
N = 20 #number of k-points for discretization of BZ: NxNxN
eta = 0.003

#loc = 'Wannier90/Y2Ir2O7/model_0.85/'
#mu = 5.019371624158303
loc = 'Wannier90/Y2Ir2O7/model_pm/'
mu = 4.959327058720621 
#loc = 'Wannier90/Y2Ir2O7/model_1.2/'
#mu = 4.996462696789173
#loc = 'Wannier90/Y2Ir2O7/model_1.2_pm/'
#mu = 4.9035310822964195
#loc = 'Wannier90/Y2Ir2O7/model_1.96/'
#mu = 4.9257963699658465
# loc = 'Wannier90/Y2Ir2O7/model_1.96_pm/'
# mu = 4.779956441230761

#integrated
q_biglist_val = [[0, [[0.0, 0.0, 0.0]]], [1, [[0.0, 0.0, 0.10471975511965977]]], [2, [[0.0, 0.0, 0.20943951023931953]]]]
q_num = len(q_biglist_val) #number of q-points

#importing Hamiltonian
latHam_data = np.genfromtxt(loc+'ham_wannier90_hr_Y2Ir2O7.dat')
L = len(latHam_data)
list_r = [[latHam_data[[i]][0][0], latHam_data[[i]][0][1], latHam_data[[i]][0][2]] for i in range(L)] #components of the position vectors in terms of the lattice vectors
list_nm = [[int(latHam_data[[i]][0][3]-1), int(latHam_data[[i]][0][4])-1] for i in range(L)] #orbital indices n and m (notation of orbitals 0-15)
list_re_im = [latHam_data[[i]][0][5]+1j*latHam_data[[i]][0][6] for i in range(L)] #hopping parameters in real space from Wannier Hamiltonian

weights_data = np.genfromtxt(loc+'nrpts_wannier90_hr_Y2Ir2O7.dat', delimiter='\n', dtype=str) #importing weights from Wannier90 output
weights_data_str = [str(t).split() for t in weights_data] 
c_weights_list = [1/int(item) for sublist in weights_data_str for item in sublist]  #c_weights_list is a flat list containing 1/weights
w = len(c_weights_list)

wfcenters_data = np.genfromtxt(loc+'wfcenters_wannier90_hr_Y2Ir2O7.dat') #centers of Wannier functions from Wannier90 output 
orb = len(wfcenters_data)
wann_R = [[wfcenters_data[[i]][0][0], wfcenters_data[[i]][0][1], wfcenters_data[[i]][0][2]] for i in range(orb)]

list_r_ar=np.array(list_r)
list_nm_ar=np.array(list_nm)
wann_R_ar=np.array(wann_R)
c_weights_list_ar=np.array(c_weights_list)
list_re_im_ar=np.array(list_re_im)

#Here we make k-biglist
#we first sample k-biglist (-pi, pi) - paralellopiped
k_val = np.arange(-np.pi, np.pi, 2*np.pi/N)
k_biglist = [[k_val[a]+1.37*10**(-12), k_val[b]-1.29*10**(-12), k_val[c]-1.13*10**(-12)] for a in range(len(k_val)) for b in range(len(k_val)) for c in range(len(k_val))]

@njit
def Ham_calc_nb(w,lk,m_biglist,list_r,orb,list_nm,wann_R,c_weights_list,list_re_im):
    Ham = np.zeros((orb, orb, lk), dtype=nb.c16)
    for i in range(w):
        list_r_orb=list_r[orb*orb*i]
        c_exp = np.exp(1j*((m_biglist*list_r_orb).sum(1)))
        for o in range(orb**2):
            row = (orb**2)*i+o
            n = list_nm[row][0]
            m = list_nm[row][1]
            # R_nm = np.array([(wann_R[n][0]-wann_R[m][0]), (wann_R[n][1]-wann_R[m][1]), (wann_R[n][2]-wann_R[m][2])])

            #calculating Hamiltonian in momentum space
            c= c_weights_list[i]*list_re_im[row]
            Ham[n][m] += c*c_exp #*np.exp(-1j*((m_biglist*R_nm).sum(1)))
    return Ham

#parallel running
def calculate(q_biglist_val):
    qvaluestr = str(q_biglist_val[0]) #gives the index of q-vector
    q_biglist = q_biglist_val[1] #gives the q-vector
    q_biglist_0 = [[0,0,0]] #we separatelly calculate for G=(0,0,0) point
    
    #making a list of k+q values in format (q_biglist)(k_biglist)
    kq_biglist = [[[q_biglist[q][0]+k_biglist[k][0], q_biglist[q][1]+k_biglist[k][1], q_biglist[q][2]+k_biglist[k][2]] for k in range(len(k_biglist))] for q in range(len(q_biglist))]
    #and the same thing for list q_biglist_0 (to have the same shape as kq_biglist for running it in the same way)
    kq_biglist_0 = [k_biglist for q in range(1)]    

    #calculating energies and eigenvectors - this function in general depends only on kq_biglist function (all information are here already), but for parallel running on cluster, it is defined also as a function of q_biglist    
    def energy(q_biglist,kq_biglist):
        all_eiglist = []
        all_eigvectors = []

        for q in range(0, len(q_biglist)):

            m_biglist=kq_biglist[q]

            eiglist = []
            eigvectors = []
            m_biglist_ar=np.array(m_biglist)

            lk=len(m_biglist)
            Ham=Ham_calc_nb(w,lk,m_biglist_ar,list_r_ar,orb,list_nm_ar,wann_R_ar,c_weights_list_ar,list_re_im_ar)

            #diagonalizing Hamiltonian for each m-value from our list separately                    
            for p in range(len(m_biglist)):
                Ham0=Ham[:,:,p]
                eig, eigv = la.eigh(Ham0)        

                eiglist.append(np.real(eig))
                eigvectors.append(eigv)

            all_eiglist.append(eiglist)
            all_eigvectors.append(eigvectors)
            
        return (all_eiglist, all_eigvectors)

    start_time = time.time()
    energylist = energy(q_biglist,kq_biglist)
    gc.collect()
    energylist1 = energy(q_biglist_0, kq_biglist_0)
    gc.collect()
    
    #eigenenergies and eigenvectors: ener & ener_v
    ener = energylist[0] #shape (q_biglist)(k_biglist)(orb)
    ener_v = energylist[1] #shape (q_biglist)(k_biglist)(orb)(orb)
    #np.save('results/energies/XWGL_energies_N='+str(len(k_val))+'_kbT='+str(kbT)+'_omega='+str(omega)+'_q='+str(q_biglist_val[0])+'.npy', ener)
    #np.save('results/energies/XWGL_energies_N='+str(N)+'_q='+str(q_biglist_val[0])+'.npy', ener)
    #np.save('fullA_test_ener_v_N='+str(N)+'_q='+str(q_biglist_val[0])+'.npy', ener_v)

    #eigenenergies and eigenvectors for q=0
    ener0 = energylist1[0][0] #shape (k_biglist)(orb)
    ener_v0 = energylist1[1][0] #shape (k_biglist)(orb)(orb)   
    print('Eigenproblem is solved.')

    del energylist
    del energylist1
    gc.collect()
    #to access eigenvectors as (bands)(components), one has to transpose them, as the python output for eigenvectors is (components)(bands)
    ener_v0_transp = [np.transpose(vec) for vec in ener_v0] #for q=0 the shape is (k_biglist, orb, orb)
    ener_v_transp = [[np.transpose(vec) for vec in vec_q] for vec_q in ener_v] #for all q: (q_biglist, k_biglist, orb, orb)

    #flat list of energies for q=0: (k_biglist, orb) to (k_biglist*orb)
    ener0_k_m = [item_k_n for ener0_k in ener0 for item_k_n in ener0_k]
    #flat list of energies for other q values: (q_biglist, k_biglist, orb) to (q_biglist*k_biglist*orb)
    ener_q_k_n = [item_q_k_n for ener_q in ener for ener_q_k in ener_q for item_q_k_n in ener_q_k]

    #fermi function - list of length (k_biglist*orb)
    nf0_k_m = [1/(np.exp((eig0_k_m-mu)/kbT)+1) for eig0_k_m in ener0_k_m]
    nf_q_k_n = [1/(np.exp((eig_q_k_n-mu)/kbT)+1) for eig_q_k_n in ener_q_k_n]
    
    del ener
    del ener_v
    del ener0
    del ener_v0
    gc.collect()
    #calculating the expression for susceptibility: (nf0_km - nf_qkn)/(eig0_km-eig_qkn)
    fermi_f_div_list = []
    for q in range(len(q_biglist)):
        for k in range(len(k_biglist)):
            for n in range(orb):
                index_qkn = q*len(k_biglist)*orb + k*orb + n
                eig_qkn = ener_q_k_n[index_qkn]
                nf_qkn = nf_q_k_n[index_qkn]
                for m in range(orb):
                    index0_km = k*orb+m            
                    eig0_km = ener0_k_m[index0_km]
                    nf0_km = nf0_k_m[index0_km]         

                    #way 1
                    fermi_f_div = (nf0_km - nf_qkn)/(eig0_km-eig_qkn+omega+1j*eta)
                    
                    #way 2
                    #if np.abs(eig_qkn-eig0_km)<0.00000001:
                    #    fermi_f_div = -(1/kbT)*(nf_qkn**2)*np.exp((eig_qkn-mu)/kbT)
                    #    #equivalent to fermi_f_div = (1/4*kbT)*(1/(np.cosh((eig_qkn-mu)/(2*kbT))**2))
                    #else:
                    #    fermi_f_div = (nf0_km - nf_qkn)/(eig0_km-eig_qkn)

                    fermi_f_div_list.append(fermi_f_div) 
    fermi_f_div_list_in=np.array(fermi_f_div_list)
    del fermi_f_div_list   
    gc.collect()

    # #CHARGE SUSC    
    # #for q=0, we will need the product (u_b^n(k))^*u_d^n(k): list(k_biglist*orb*orb*orb)
    # ener_v0_k_m_c_d_in = np.array([item_c*np.conj(item_d) for ener_v0_k in ener_v0_transp for ener_v0_k_n in ener_v0_k for item_c in ener_v0_k_n for item_d in ener_v0_k_n])
    # #for q\neq0, we will need the product (u_a^n(k+q))^*u_c^n(k+q): list(q_biglist*k_biglist*orb*orb*orb)
    # ener_v_q_k_n_a_b_in = np.array([item_a*np.conj(item_b) for ener_v_q in ener_v_transp for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n for item_b in ener_v_q_k_n])                        

    #SPIN SUSC - all channels
    ener_v0_transp = np.array(ener_v0_transp)
    ener_v_transp = np.array(ener_v_transp)
    mask_ener_v0_up = np.zeros((len(k_biglist), orb, orb))
    mask_ener_v0_up[:, :, :orb // 2] = 1.  # Set ones in the first half of the last axis
    mask_ener_v0_dn = np.zeros((len(k_biglist), orb, orb))
    mask_ener_v0_dn[:, :, orb // 2:] = 1.  # Set ones in the second half of the last axis
    ener_v0_transp_up = ener_v0_transp * mask_ener_v0_up
    ener_v0_transp_dn = ener_v0_transp * mask_ener_v0_dn
    mask_ener_v_up = np.zeros((len(q_biglist), len(k_biglist), orb, orb))
    mask_ener_v_up[:, :, :orb // 2] = 1.  # Set ones in the first half of the last axis
    mask_ener_v_dn = np.zeros((len(q_biglist), len(k_biglist), orb, orb))
    mask_ener_v_dn[:, :, orb // 2:] = 1.  # Set ones in the second half of the last axis
    ener_v_transp_up = ener_v_transp * mask_ener_v_up
    ener_v_transp_dn = ener_v_transp * mask_ener_v_dn
 
    ener_v0_k_m_b_d_uu = np.array([item_b*np.conj(item_d) for ener_v0_k in ener_v0_transp_up for ener_v0_k_n in ener_v0_k for item_b in ener_v0_k_n for item_d in ener_v0_k_n])
    ener_v0_k_m_b_d_dd = np.array([item_b*np.conj(item_d) for ener_v0_k in ener_v0_transp_dn for ener_v0_k_n in ener_v0_k for item_b in ener_v0_k_n for item_d in ener_v0_k_n])
    ener_v_q_k_n_a_c_uu = np.array([item_a*np.conj(item_c) for ener_v_q in ener_v_transp_up for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n for item_c in ener_v_q_k_n])                        
    ener_v_q_k_n_a_c_dd = np.array([item_a*np.conj(item_c) for ener_v_q in ener_v_transp_dn for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n for item_c in ener_v_q_k_n])                            
    v_up = np.zeros(orb)
    v_up[:orb // 2] = 1.  # Set ones in the first half of the last axis
    v_dn = np.zeros(orb)
    v_dn[orb // 2:] = 1.  # Set ones in the second half of the last axis
    ener_v0_k_m_b_d_ud = np.array([item_b*np.conj(item_d) for ener_v0_k in ener_v0_transp for ener_v0_k_n in ener_v0_k for item_b in ener_v0_k_n*v_up for item_d in ener_v0_k_n*v_dn])
    ener_v0_k_m_b_d_du = np.array([item_b*np.conj(item_d) for ener_v0_k in ener_v0_transp for ener_v0_k_n in ener_v0_k for item_b in ener_v0_k_n*v_dn for item_d in ener_v0_k_n*v_up])
    ener_v_q_k_n_a_c_ud = np.array([item_a*np.conj(item_c) for ener_v_q in ener_v_transp for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n*v_up for item_c in ener_v_q_k_n*v_dn])                        
    ener_v_q_k_n_a_c_du = np.array([item_a*np.conj(item_c) for ener_v_q in ener_v_transp for ener_v_q_k in ener_v_q for ener_v_q_k_n in ener_v_q_k for item_a in ener_v_q_k_n*v_dn for item_c in ener_v_q_k_n*v_up])                        
 
    del ener_v0_transp
    del ener_v_transp
    del ener_v0_transp_up
    del ener_v0_transp_dn
    del ener_v_transp_up
    del ener_v_transp_dn
    del ener0_k_m
    del ener_q_k_n
    del nf0_k_m
    del nf_q_k_n
    gc.collect()
    #here we do the summmation over k, m and n and calculate susceptibility as function of q and (a, b, c, d)
    k_biglist_in=np.array(k_biglist)

    time2 = round((time.time() - start_time), 2)

    @njit
    def susceptibility_flat_n(a, b, c, d, k_biglist1, ener_v_q_k_n_a_c1, ener_v0_k_m_b_d1, fermi_f_div_list1, orb1):
        susc = 0 + 0j
        q = 0
        lk = len(k_biglist1)

        for k in range(lk):
            for n in range(orb1):
                una_unc = ener_v_q_k_n_a_c1[q * orb1 * orb1 * orb1 * lk + k * orb1 * orb1 * orb1 + n * orb1 * orb1 + a * orb1 + c]
                for m in range(orb1):
                    umb_umd = ener_v0_k_m_b_d1[k * orb1 * orb1 * orb1 + m * orb1 * orb1 + b * orb1 + d]                  
                    fermi_f_div = fermi_f_div_list1[q * orb1 * orb1 * lk + k * orb1 * orb1 + n * orb1 + m]
                    susc += una_unc * umb_umd * fermi_f_div
        return np.array([-susc / lk])

    
    #abba
    # susc = 0+0.j
    susc_uudd = 0+0.j
    susc_dduu = 0+0.j
    susc_uuuu = 0+0.j
    susc_dddd = 0+0.j
    susc_uddu = 0+0.j
    susc_duud = 0+0.j
    susc_tr =  0+0.j
    susc_lo =  0+0.j
    susc_ch =  0+0.j
    # print('time needed for calculating all channels of susceptibility for', len(q_biglist), 'q-values:')
    # start_time = time.time()
    for a in range(orb):
        for b in range(orb):
            # susc_abba= susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_d_in, ener_v0_k_m_b_c_in, fermi_f_div_list_in, orb)
            # susc+=susc_abba
            uudd = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_uu, ener_v0_k_m_b_d_dd, fermi_f_div_list_in, orb)
            dduu = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_dd, ener_v0_k_m_b_d_uu, fermi_f_div_list_in, orb)
            uuuu = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_uu, ener_v0_k_m_b_d_uu, fermi_f_div_list_in, orb)
            dddd = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_dd, ener_v0_k_m_b_d_dd, fermi_f_div_list_in, orb)
            uddu = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_ud, ener_v0_k_m_b_d_du, fermi_f_div_list_in, orb)
            duud = susceptibility_flat_n(a, b, b, a, k_biglist_in, ener_v_q_k_n_a_c_du, ener_v0_k_m_b_d_ud, fermi_f_div_list_in, orb)

            susc_uudd += uudd
            susc_dduu += dduu
            susc_uuuu += uuuu
            susc_dddd += dddd
            susc_uddu += uddu
            susc_duud += duud

            susc_tr += (uudd + dduu)
            susc_lo += (uuuu + dddd - uddu - duud)
            susc_ch += (uuuu + dddd + uddu + duud)

    # time2 = round((time.time() - start_time), 2)
    # print(time2, 'sec')
    # print('DONE')

    # print('res', [susc_tr, susc_uudd, susc_dduu, susc_uuuu, susc_dddd])

    np.savetxt('results/all_channels_N='+str(len(k_val))+'_kbT='+str(kbT)+'_omega='+str(omega)+'_q='+str(q_biglist_val[0])+'.dat', [susc_tr, susc_lo, susc_ch, susc_uudd, susc_dduu, susc_uuuu, susc_dddd, susc_uddu, susc_duud])
    return 1

if __name__ == '__main__':
    p = Pool(q_num)
    print(p.map(calculate, q_biglist_val))

EOF
        # # Execute the generated Python file
        # python "Lindhard_Y2Ir2O7_omega=${omega}_kbT=${kbT}.py"
    done
done

# Exit the script
exit 0