module MakeGlassBox

using HDF5
using StaticArrays
using Statistics
using Random
using LinearAlgebra
using SpecialFunctions
using Printf
using Parameters
#using .Threads
using PyPlot

using OctreeBH

export generate_ICfile_glass_box, plot_convergence
export Params

function vec2svec(vec::Matrix{T}) where {T}
    ndim = size(vec,1)
    svec = [SVector{ndim,T}(vec[:,i]) for i in 1:size(vec,2)]
end
function mat2smat(mat::Array{T,3}) where {T}
    ndim = size(mat,1)
    smat = [SMatrix{ndim,ndim,T}(mat[:,:,i]) for i in 1:size(mat,3)]
end
const KERNELCONST = (16.0/pi)
@inline ramp(x) = max(0, x);
@inline function kernel_cubic(x::T) where {T}
    return (KERNELCONST * (ramp(1.0 - x)^3 - 4.0 * ramp(0.5 - x)^3))
end

const SOLARMASS = 1.989e+43
const PROTONMASS = 1.6726e-24
const XH = 0.71
const BOLTZMANN = 1.38e-16
const GAMMA = 5.0 / 3


@with_kw struct Params{NDIM,T}

    boxsize::T 

    nH::T = 0.
    T_mu::T = 0.

    UnitLength_in_cm::T = 3.08568e+21
    UnitMass_in_g::T = 1.989e+43
    UnitVelocity_in_cm_per_s::T = 1e5
    u0::T = T_mu * BOLTZMANN / (GAMMA-1.0) / PROTONMASS / UnitVelocity_in_cm_per_s^2
    Mgas_tot::T = nH * PROTONMASS * (boxsize*UnitLength_in_cm)^3 / XH / UnitMass_in_g

    seed::Int64 = 0

    #Ngas is always known s.t. we can calculate h0 & Lint below
    #OTOH, mgas can be zero if Ngas is user input
    mgas::T = 0.0
    Ngas::Int64 = (mgas > 0.) ? round(Int64, Mgas_tot / mgas) : Ngas

    filename::String = ""

    ms::T = 1.0
    flag_plot_part::Bool = false
    flag_plot_conv::Bool = false

    #NDIM::Int64 = 3
    geo_fac::T = pi^(0.5*NDIM)/gamma(0.5*NDIM+1)

    Nngb::T = (100.0)^(NDIM/3.0)

    unit_boxsize = 1.0
    Lint::T = unit_boxsize / Ngas^(1.0/NDIM) #inter-particle spacing
    h0::T = Lint * (Nngb / geo_fac)^(1.0/NDIM)

    Nstep::Int64 = 30
    Njump::Int64 = 3
    Nidx::Int64 = NDIM-1
    A0::T = 1.0
    alpha::T = 0.1 / sqrt(Lint)    
end

function calc_force!(force,Nngb_true,Nngb_weight,X,V,tree,par::Params{NDIM,T}) where {NDIM,T}
    
    @unpack boxsize, mgas, geo_fac, Lint, h0, Nidx, A0, alpha = par
    #NDIM=3
    unit_boxsize = 1.0
    boxsizes = SVector{NDIM}(ones(NDIM)) .* unit_boxsize
    
    Threads.@threads for i in eachindex(X)
        force[i] = SVector{NDIM}(zeros(NDIM))
        #idx_ngbs = get_scatter_ngb_tree(X[i], tree, boxsizes)
        idx_ngbs = get_gather_ngb_tree(X[i], h0, tree, boxsizes)
        Nngb_weight[i] = 0.0
        for k in eachindex(idx_ngbs)
            j = idx_ngbs[k]
            if i==j continue end
            dx = nearest.(X[i] - X[j], boxsizes) ./ Lint #from j to i, positive = respulsive!
            dr = norm(dx)
            force[i] += (A0/dr^Nidx .* (dx./dr) - alpha .* V[i])
            Wij = kernel_cubic(dr*Lint/h0)
            Nngb_weight[i] += Wij
        end
        Nngb_weight[i] *= geo_fac
        Nngb_true[i] = length(idx_ngbs)
    end
end

function make_glass_box(X, V, par::Params{NDIM,T}) where {NDIM,T}
    
    @unpack boxsize, Ngas, h0, Lint, ms, flag_plot_part, flag_plot_conv, geo_fac, Nngb, Nstep, Njump, Nidx = par
    #NDIM=3
    mass = mass_H2 = mass_CO = zeros(Ngas)

    unit_boxsize = T(1)
    boxsizes = SVector{NDIM,T}(ones(NDIM)) .* unit_boxsize #use boxsize=1 here!!! otherwise double-counting!!!!!!    
    center = boxsizes .* T(0.5)

    hsml = ones(Ngas) .* h0

    part = [Data{NDIM,T}(X[i], i, hsml[i], mass[i]) for i in 1:Ngas]
    tree = buildtree(part, center, boxsizes);

    Nngb_true = zeros(Ngas)
    Nngb_weight = zeros(Ngas)
    force = [SVector{NDIM}(zeros(NDIM)) for _ in eachindex(X)]
    fac_dt = T(0.3)

    std_dNngb_true = zeros(Nstep+1)
    std_dNngb_weight = zeros(Nstep+1)
    calc_force!(force,Nngb_true,Nngb_weight,X,V,tree,par)

    std_dNngb_true[1] = std(Nngb_true .- Nngb) / Nngb
    std_dNngb_weight[1] = std(Nngb_weight .- Nngb) / Nngb

    dt = fac_dt * minimum( @. sqrt(Lint / norm(force)) )

    @. V = V + 0.5*dt * force
    for i in 1:Nstep
        println("istep = ", i, "  Nstep = ", Nstep)
        if flag_plot_part && i%Njump==0 
            fig, ax = subplots(1, 1, figsize=(6, 6))
            ax.plot(getindex.(X,1), getindex.(X,2), ".", ms=ms)
            ax.set_aspect("equal")
            ax.axis([0,1,0,1])
            fig.tight_layout()
            fig.savefig("ic_relax_"*string(div(i,Njump))*".png")
            clf()
        end
        @. X = X + dt * V    
        for j in eachindex(X)
            X[j] -= convert.(T, X[j] .> boxsizes) .* boxsizes
            X[j] += convert.(T, X[j] .< 0.0) .* boxsizes
        end

        part = [Data{NDIM,T}(X[i], i, hsml[i], mass[i]) for i in 1:Ngas]
        tree = buildtree(part, center, boxsizes);
    
        calc_force!(force,Nngb_true,Nngb_weight,X,V,tree,par)
        @. V = V + dt * force
        dt = fac_dt * minimum( @. sqrt(Lint / norm(force)) )
        
        std_dNngb_true[i+1] = std(Nngb_true .- Nngb) / Nngb
        std_dNngb_weight[i+1] = std(Nngb_weight .- Nngb) / Nngb
    end
    #return Nngb_true, std_dNngb_true, Nngb_weight, std_dNngb_weight
    if flag_plot_conv == true
        plot_convergence(std_dNngb_true, std_dNngb_weight, Nngb_true, Nngb_weight, Nngb)
    end
end

function generate_ICfile_glass_box(par::Params{NDIM,T}) where{NDIM,T}

    @unpack nH, T_mu, boxsize, Ngas, mgas, Mgas_tot, u0, filename, seed = par
    
    mgas = Mgas_tot / Ngas

    @show mgas, Ngas, boxsize
    id_gas = collect(Int32, 1:Ngas)
    mass = ones(Ngas) .* mgas
    u = ones(Ngas) .* u0;
    #hsml = ones(Ngas) .* h0

    ########## prepare glass configuration ##########
    unit_boxsize = 1.0
    #pos = rand(MersenneTwister(seed), NDIM, Ngas) .* unit_boxsize;
    pos = zeros(NDIM, Ngas);
    vel = zeros(NDIM, Ngas);

    Random.seed!(seed);
    X = [SVector{NDIM,T}(rand(NDIM)) for _ in 1:Ngas] .* unit_boxsize
    V = [SVector{NDIM,T}(zeros(NDIM)) for _ in 1:Ngas]
    #Nngb_true, std_dNngb_true, Nngb_weight, std_dNngb_weight = 
    @time make_glass_box(X,V,par);

    #save back to pos and scale to the actual boxsize
    for i in eachindex(X)
        pos[:,i] = X[i] .* boxsize;  #do this only to positions; velocities remain zero
    end
    

    if filename == ""
        filename = "ics" 
        if nH > 0.
            filename *= "_nH" * @sprintf("%.0e", nH) * "_T" * @sprintf("%.0e", T_mu)
        else
            filename *= "_Mtot" * @sprintf("%.0e", Mgas_tot) * "_u" * @sprintf("%.0e", u0)
        end
        filename *= "_boxsize" * @sprintf("%.2e", boxsize) * 
                    "_Ngas" * @sprintf("%.0e", Ngas) *
                    "_mgas" * @sprintf("%.1e", mgas) * ".hdf5"
    end

    ########## write to file ##########
    save_gadget_ics(filename, pos, vel, id_gas, mass, u, Ngas, boxsize)

    return pos, vel, id_gas, mass, u, Ngas, boxsize    
end

function save_gadget_ics(filename, pos, vel, id_gas, mass, u, Ngas, boxsize)
    T = Float32

    fid=h5open(filename,"w")

    grp_head = create_group(fid,"Header");
    attributes(fid["Header"])["NumPart_ThisFile"]       = Int32[Ngas, 0, 0, 0, 0, 0]
    attributes(fid["Header"])["MassTable"]              = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    attributes(fid["Header"])["Time"]                   = 0.0
    attributes(fid["Header"])["Redshift"]               = 0.0
    attributes(fid["Header"])["Flag_Sfr"]               = 0
    attributes(fid["Header"])["Flag_Feedback"]          = 0
    attributes(fid["Header"])["NumPart_Total"]          = Int32[Ngas, 0, 0, 0, 0, 0]
    attributes(fid["Header"])["Flag_Cooling"]           = 0
    attributes(fid["Header"])["NumFilesPerSnapshot"]    = 1
    attributes(fid["Header"])["BoxSize"]                = boxsize
    attributes(fid["Header"])["Omega0"]                 = 0.27
    attributes(fid["Header"])["OmegaLambda"]            = 0.73
    attributes(fid["Header"])["HubbleParam"]            = 1.0
    attributes(fid["Header"])["Flag_StellarAge"]        = 0
    attributes(fid["Header"])["Flag_Metals"]            = 0
    attributes(fid["Header"])["NumPart_Total_HighWord"] = UInt32[0,0,0,0,0,0]
    attributes(fid["Header"])["flag_entropy_instead_u"] = 0
    attributes(fid["Header"])["Flag_DoublePrecision"]   = 0
    attributes(fid["Header"])["Flag_IC_Info"]           = 0
    #attributes(fid["Header"])["lpt_scalingfactor"] = 

    grp_part = create_group(fid,"PartType0");
    h5write(filename, "PartType0/Coordinates"   , T.(pos))
    h5write(filename, "PartType0/Velocities"    , T.(vel))
    h5write(filename, "PartType0/ParticleIDs"   , id_gas)
    h5write(filename, "PartType0/Masses"        , T.(mass))
    h5write(filename, "PartType0/InternalEnergy", T.(u))

    close(fid)
end

function plot_convergence(std_dNngb_true::Vector{T}, std_dNngb_weight::Vector{T}, Nngb_true::Vector{T}, Nngb_weight::Vector{T}, Nngb::T) where {T}
    ##### convergence test
    fig = figure("convergence test",figsize=(9,4))
    subplot(121)
    plot(collect(Int32,1:length(std_dNngb_true)), std_dNngb_true, "-", label="true")
    plot(collect(Int32,1:length(std_dNngb_weight)), std_dNngb_weight, "-", label="weighted")
    ylabel("std(dNngb / Nngb)")
    ylim(0,0.22)
    legend(loc="best", frameon=false)

    subplot(122)
    hist(mean( (Nngb ./ Nngb_true ) )   .* Nngb_true  ./Nngb, bins=30, label="true"    , histtype="step", normed=true, range=[0.85,1.15])
    hist(mean( (Nngb ./ Nngb_weight ) ) .* Nngb_weight./Nngb, bins=30, label="weighted", histtype="step", normed=true, range=[0.85,1.15])
    ylabel("Nngb_true / Nngb (final)")
    legend(loc="best", frameon=false)

    tight_layout()
    Ngas = length(Nngb_true)
    savefig("dNngb_converge_"*string(Ngas)*".png")
    0
end

end #module MakeGlassBox

