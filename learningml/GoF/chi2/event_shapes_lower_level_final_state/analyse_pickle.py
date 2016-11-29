import pickle

def N_nonzeroone (a):
	return len(a)-sum(x == 0 for x in a)- sum(x == 1 for x in a)

bin_boundaries_dict = pickle.load( open( "bin_boundaries_dict.p", "rb" ) )

p0_list		= []
theta0_list 	= []
p1_list 	= []
theta1_list	= []
phi1_list	= []
nJetsS_list	= []
Mult_list	= []
pSum_list	= []



for bin_key, bin_boundary in bin_boundaries_dict.items():
	if '5' in bin_key: 
		print bin_key, bin_boundary
		p0_list.append(bin_boundary[0][0])
		theta0_list.append(bin_boundary[1][0])
		p1_list.append(bin_boundary[2][0])
		theta1_list.append(bin_boundary[3][0])
		phi1_list.append(bin_boundary[4][0])
		nJetsS_list.append(bin_boundary[5][0])
		Mult_list.append(bin_boundary[6][0])
		pSum_list.append(bin_boundary[7][0])

                p0_list.append(bin_boundary[0][1])
                theta0_list.append(bin_boundary[1][1])
                p1_list.append(bin_boundary[2][1])
                theta1_list.append(bin_boundary[3][1])
                phi1_list.append(bin_boundary[4][1])
                nJetsS_list.append(bin_boundary[5][1])
                Mult_list.append(bin_boundary[6][1])
                pSum_list.append(bin_boundary[7][1])


p0_nonzeroone= N_nonzeroone(p0_list)
theta0_nonzeroone= N_nonzeroone(theta0_list)
p1_nonzeroone= N_nonzeroone(p1_list)
theta1_nonzeroone= N_nonzeroone(theta1_list)
phi1_nonzeroone= N_nonzeroone(phi1_list)
nJetsS_nonzeroone= N_nonzeroone(nJetsS_list)
Mult_nonzeroone= N_nonzeroone(Mult_list)
pSum_nonzeroone= N_nonzeroone(pSum_list)

print p0_nonzeroone, theta0_nonzeroone, p1_nonzeroone, theta1_nonzeroone, phi1_nonzeroone, nJetsS_nonzeroone, Mult_nonzeroone, pSum_nonzeroone
