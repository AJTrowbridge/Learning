# feed forward monovariate regression neural network prototype

using LinearAlgebra, Statistics, Distributions

σ(x) = 1 / (1 + exp(-x))
dσ(y) = y*(1 - y)
logit(x) = log(x / (1 - x))

struct Dataset
    x::Vector{Float64}
    y::Vector{Float64}
end

mutable struct Neuron
    w::Vector{Float64}
    b::Float64
    x::Float64
    y::Float64
    Neuron(n::Int) = new(rand(winitdistr, n), 1.0, 0.0, 0.0)
end

mutable struct Brain
    xvec::Vector{Float64}
    mind::Matrix{Neuron}
    outl::Vector{Neuron}
    yvec::Vector{Float64}
    err::Vector{Float64}
    Δs::Vector{Vector{Float64}}
end

function datagen(f::Function, rng::StepRangeLen)
    batch = []
    dx = 0.1
    for set = 1:batchsize
        x = rand(rng, samplsize)
        y = f.(x)
        dat = Dataset(x,y)
        push!(batch, dat)
    end
    batch
end

function init()
    mind = Matrix{Neuron}(undef, D, L)
    for l = 1:size(mind, 2)
        for j = 1:size(mind, 1)
            l == 1 ? mind[j,l] = Neuron(samplsize) : mind[j,l] = Neuron(D)
        end
    end
    outl = [Neuron(D) for a = 1:samplsize]
    Δs = [[zeros(D) for l = 1:L]; [zeros(samplsize)]]
    Brain([], mind, outl, [], [], Δs)
end

function forward!(nn::Brain)
    for l = 1:size(nn.mind, 2)
        for i = eachindex(nn.mind[:,l])
            l == 1 ? nn.mind[i,l].x = dot(nn.mind[i,l].w, nn.xvec) + nn.mind[i,l].b :
                     nn.mind[i,l].x = dot(nn.mind[i,l].w, [n.y for n in nn.mind[:,l-1]]) + nn.mind[i,l].b
            nn.mind[i,l].y = σ(nn.mind[i,l].x)
        end
    end
    for j = eachindex(nn.outl)
        nn.outl[j].x = dot(nn.outl[j].w, [n.y for n in nn.mind[:,end]]) + nn.outl[j].b
        nn.outl[j].y = σ(nn.outl[j].x)
    end
    nn.yvec = [neuron.y for neuron in nn.outl]
end

function collectΔs!(nn::Brain)
    for j = eachindex(nn.outl)
        Δj = nn.err[j]*dσ(nn.outl[j].y)
        nn.Δs[L+1][j] += Δj
    end
    for l = L:-1:1
        for j = eachindex(nn.mind[:,l])
            l == L ? Δj = dσ(nn.mind[j,l].y)*dot([nn.outl[k].w[j] for k = 1:size(nn.outl, 1)], nn.Δs[L+1]) :
                     Δj = dσ(nn.mind[j,l].y)*dot([nn.mind[k,l+1].w[j] for k = 1:size(nn.mind[:,l+1], 1)], nn.Δs[l+1])
            nn.Δs[l][j] += Δj
        end
    end
end

function passΔs!(nn::Brain)
    for l = 1:L
        for j = eachindex(nn.mind[:,l])
            Δj = nn.Δs[l][j]
            for i = eachindex(nn.mind[j,l].w)
                l == 1 ? nn.mind[j,l].w[i] += -η*nn.xvec[i]*Δj :
                         nn.mind[j,l].w[i] += -η*nn.mind[i,l-1].y*Δj
            end
            nn.mind[j,l].b += -η*Δj
        end
    end
    for j = eachindex(nn.outl)
        Δj = nn.Δs[L+1][j]
        for i = eachindex(nn.outl[j].w)
            nn.outl[j].w[i] += -η*nn.mind[i,end].y*Δj
        end
        nn.outl[j].b += -η*Δj
    end
    nn.Δs = [[zeros(D) for l = 1:L]; [zeros(samplsize)]]
end

function train!(nn::Brain, dat::Dataset)
    nn.xvec = dat.x
    forward!(nn)
    nn.err = nn.yvec - dat.y
    collectΔs!(nn)
    passΔs!(nn)
end

function report(itr::Int, errs::Vector, io::IO)
    avgerr = mean([0.5*norm(err)^2 for err in errs])
    println(io, "$(itr) $(avgerr)")
    println("itr    = $(itr)")
    println("avgerr = $(avgerr)\n")
end

function graph(nn::Brain, fn::Function, rng::StepRangeLen)
    netio = open("net.dat", "w")
    solio = open("sol.dat", "w")
    ubnd = maximum(rng)
    lbnd = minimum(rng)
    step = (ubnd - lbnd) / 100
    for x = lbnd:step:ubnd
        yfn = fn(x)
        nn.xvec = [x]
        forward!(nn)
        ynn = nn.yvec[1]
        println(netio, "$(x) $(ynn)")
        println(solio, "$(x) $(yfn)")
    end
end

const winitdistr = Normal(0, 2)
const batchsize  = 40
const samplsize  = 1
const maxitr     = 500000
const L          = 1
const D          = 4
const η          = 0.015

function main()
    fn(x) = sin(2π*x)^2
    range = 0:1e-10:1
    brain = init()
    io    = open("error.dat", "w")
    errs  = []
    for itr = 1:maxitr
        batch = datagen(fn, range)
        for dataset in batch train!(brain, dataset) end
        push!(errs, brain.err)
        probe = mod(itr, 250) == 0
        probe ? begin report(itr, errs, io); errs = [] end : continue
    end
    graph(brain, fn, range)
end

@time main()




