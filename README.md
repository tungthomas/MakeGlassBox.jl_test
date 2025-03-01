# MakeGlassBox.jl
generate a glass-like particle distribution in a cubic box

Pei: Add a Project.toml to download


# Quick Start
```
using .MakeGlassBox
NDIM = 2
par = Params{NDIM,Float64}(nH=1.0, T_mu=1e4, boxsize=0.1, Ngas=1000, flag_plot_part=true, ms=3)
generate_ICfile_glass_box(par)
```

![make_glass_2D_N1000](https://user-images.githubusercontent.com/23061774/153781773-a56accd5-e940-405c-b66f-b9ddc5880712.gif)
