# Lanczos Diagonalization of 1D Transverse Field Ising model with random longitudinal fields
using LinearAlgebra, SparseArrays, Arpack, PyPlot, Plots, JLD, DelimitedFiles

# Constructs the Hamiltonian in sparse CSC format
function makeSparseHamiltonian(n,s;λ=0,seed=nothing)
    num_conf = 2^n
    if seed!=nothing
        Random.seed!(seed)
    end
    g = 2.0.*rand(Float64,n).-1.0
    basis = reverse(2 .^(0:n-1))
    rowval = repeat(1:num_conf, inner=n+1)
    colptr = Vector{Int64}(undef, num_conf*(n+1))
    data = (s-1).*ones(num_conf*(n+1))
    bits = Vector{Int64}(undef,n)

    for i in 0:num_conf-1 #0:(num_conf÷2-1) #
        bits = reverse(digits!(bits, i, base=2))
        config = bits.*2 .-1
        data[i*(n+1)+1] = -s*(sum(config[1:end-1].*config[2:end]) + config[1]*config[end]) + λ*sum(g.*config)
        #data[(num_conf-i-1)*(n+1)+1] = -data[i*(n+1)+1]
        colptr[i*(n+1)+1] = i+1
        #colptr[(num_conf-i-1)*(n+1)+1] = num_conf-i
        colptr[i*(n+1)+2:(i+1)*(n+1)] = -config.*basis .+ (i+1)
        #colptr[(num_conf-i-1)*(n+1)+2:(num_conf-i)*(n+1)] = config.*basis .+ (num_conf-i)
    end
    H = sparse(colptr,rowval,data)
    return H
end


# Finds eigenvalues for multiple system sizes in a noiseless system
function runNoiselessModel(numSs,numNs,startN;nev=2)
    Ns = startN:startN+numNs-1
    Ss = range(0.0,1.0,length=numSs)
    eigvals = Array{Float64}(undef,nev,numSs,numNs)
    for (n_ind,n) in enumerate(Ns)
        println(n)
        for i in 1:numSs
            H = makeSparseHamiltonian(n,Ss[i];λ=0,seed=nothing)
            eigvas,eigves = eigs(H;nev=nev,which=:SR)
            eigvals[:,i,n_ind] = eigvas
        end
    end
    return eigvals
end


# Finds the mean energy and ground state probabilities for fixed N, λ
function runLattice(n,λ,Ss,numSs;nev=3,nbins=20,nsamples=100,verbose=false)

    deltas = Array{Float64}(undef,2,nbins,numSs)
    deltas_sq = Array{Float64}(undef,2,nbins,numSs)
    ground_prob = Array{Float64}(undef,nbins,numSs)
    ground_prob2 = Array{Float64}(undef,nbins,numSs)

    for s_i in 1:numSs
        s = Ss[s_i]
        if verbose
            println(s_i)
        end
        for ibin in 1:nbins
            deln, deln2 = zeros(2), zeros(2)
            gp, gp2 = 0.0, 0.0
            for isample in 1:nsamples
                H = makeSparseHamiltonian(n,s;λ=λ,seed=nothing)
                eigvals,eigvecs = eigs(H;nev=nev,which=:SR,maxiter=2000)
                #deln += (eigvals[2:3] - eigvals[1:2])
                #deln2 += (eigvals[2:3] - eigvals[1:2]).^2
                deln[1]  +=  eigvals[2]-eigvals[1]
                deln[2]  +=  eigvals[3]-eigvals[1]
                deln2[1] += (eigvals[2]-eigvals[1])^2
                deln2[2] += (eigvals[3]-eigvals[1])^2
                gp  += (eigvecs[1,1]^2 + eigvecs[end,1]^2)
                gp2 += (eigvecs[1,1]^2 + eigvecs[end,1]^2)^2
            end
            deltas[:,ibin,s_i]     = deln./nsamples
            deltas_sq[:,ibin,s_i]  = deln2./nsamples #- (deln./nsamples).^2
            ground_prob[ibin,s_i]  = gp/nsamples
            ground_prob2[ibin,s_i] = gp2/nsamples #- (gp/nsamples)^2
        end
    end
    av_deltas = sum(deltas,dims=2)./nbins
    av_gp = sum(ground_prob,dims=1)./nbins
    av_deltas_sq = copy(deltas_sq)
    av_gp_sq = copy(ground_prob2)

    for i in 1:nbins
        av_deltas_sq[:,i,:] = av_deltas_sq[:,i,:] - (av_deltas.^2)[:,1,:]#(sum(deltas_sq,dims=2)./nbins)[:,1,:]
        av_gp_sq[i,:] = av_gp_sq[i,:] - (av_gp.^2)[1,:]#(sum(ground_prob2,dims=1)./nbins)[1,:]
    end

    av_deltas_sq = sum(av_deltas_sq,dims=2)./(nbins*(nbins-1))
    av_gp_sq = sum(av_gp_sq,dims=1)./(nbins*(nbins-1))

    return av_deltas, av_deltas_sq, av_gp, av_gp_sq
end

numSs = 20
Ss = range(0.0,1.0,length=numSs)

nonoise_eigvals = runNoiselessModel(numSs,13,4;nev=2)
for n_ind in 1:size(nonoise_eigvals)[3]
    PyPlot.plot(Ss,nonoise_eigvals[1,:,n_ind])
end
PyPlot.display_figs()

for n in 5:18
    t = 0.0
    for reps in 1:5
        t+= (@elapsed H = makeSparseHamiltonian(n,0.5;λ=0.5,seed=nothing))
        t+= (@elapsed eigvals,eigvecs = eigs(H;nev=3,which=:SR,maxiter=2000))
    end
    t = t/5
    println("N=",n,"; Estimated hours = ",t*20*1000/3600)
end


av_deltas, av_deltas_sq, av_gp, av_gp_sq = runLattice(10,1.0,Ss,numSs;nev=3,nbins=20,nsamples=50,verbose=true)

PyPlot.figure()
#PyPlot.errorbar(Ss,av_gp[:],yerr=sqrt.(abs.(av_gp_sq[:])))
PyPlot.plot(Ss,av_deltas[1,1,:])
PyPlot.display_figs()

directory = "workingmemory/comptoolsqmb/project/"
file1 = string("av_deltasN",string(n),"L",string(λ))
file2 = string("av_deltas_sqN",string(n),"L",string(λ))
file3 = string("av_gpN",string(n),"L",string(λ))
file4 = string("av_gp_sqN",string(n),"L",string(λ))
save(string(directory,file1,".jld"), file1, av_deltas)
save(string(directory,file2,".jld"), file2, av_deltas_sq)
save(string(directory,file3,".jld"), file3, av_gp)
save(string(directory,file4,".jld"), file4, av_gp_sq)
