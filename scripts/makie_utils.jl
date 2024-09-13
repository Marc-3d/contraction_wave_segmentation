function format_VFs( PIVs::Vector{<:Any} )
    ntps   = length( PIVs ); 
    ndims  = size( PIVs[1], 1 )
    vfsize = size( PIVs[1] )[2:end]
    vfaxes = axes( PIVs[1] )[2:end]
    output = [ zeros( Float32, ndims, vfsize... ) for t in 1:ntps ]
    for t in 1:ntps
        for c in 1:ndims
            output[t][c,vfaxes...] .= PIVs[t][c,vfaxes...]
        end
    end
    return output
end

function format_IMGs( IMGs::Array; maxI=maximum(IMGs) )
    ntps     = size( IMGs )[end]
    img_axes = axes( IMGs )[1:end-1]
    return [ min.( maxI, IMGs[img_axes...,t] ) for t in 1:ntps ]
end

function format_WVs( WVs::Array, img_size )

    fmt_WVs = [ Images.imresize( wv, img_size ) for wv in WVs ];
    [ fmt_WVs[i][ abs.( fmt_WVs[i] ) .< 0.2 ] .= NaN for i in 1:length(WVs) ]
    [ fmt_WVs[i][1,1] =  1 for i in 1:length(WVs) ]
    [ fmt_WVs[i][1,2] = -1 for i in 1:length(WVs) ]
    return fmt_WVs
end

"""
    each timepoint is assumed to be a vectorfield stored as  single array that contains all vector field
    component in the first dimension. this is how quickPIV returns vectorfields.
"""
function scrollable_vectorfield( PIVs, images; 
                                 IA=(16,16), step=(16,16), vf_subsample=(1,1),
                                 fig_width=400,
                                 fig_height=800,
                                 figsize=(fig_width,fig_height), 
                                 cmap_data=:grays,
                                 arrow_color=nothing,
                                 scale_range=0.0:0.1:2.0, 
                                 arrow_range=0.0:0.1:1.0, 
                                 max_speed=Inf,
                                 min_speed=0.0 )

    # vector fields coordinates

    xgrid = [ ( x - 1 )*step[2] + div(IA[2],2) for x in 1:size(PIVs[1],3) ]
    ygrid = [ ( y - 1 )*step[1] + div(IA[1],2) for y in 1:size(PIVs[1],2) ]

    vf_yrange = 1:vf_subsample[1]:size(PIVs[1],2)
    vf_xrange = 1:vf_subsample[1]:size(PIVs[1],3)
    xgrid_sub = [ ( x - 1 )*step[2] + div(IA[2],2) for x in vf_xrange ]
    ygrid_sub = [ ( y - 1 )*step[1] + div(IA[1],2) for y in vf_yrange ]

    GLMakie.closeall()

    begin # Observables

        # Initial value of observables, aka the first timepoint
        U_0 = PIVs[1][1,vf_yrange,vf_xrange]
        V_0 = PIVs[1][2,vf_yrange,vf_xrange]
        M_0 = sqrt.( U_0 .^ 2 .+ V_0 .^ 2 );
        s_0 = (scale_range[end]+scale_range[1])/2
        a_0 = (arrow_range[end]+arrow_range[1])/2

        # Observables
        M = Observable( M_0[:] )
        U = Observable( U_0 ./ M_0 .* M_0  )
        V = Observable( V_0 ./ M_0 .* M_0  )
        I = Observable( images[1] ); 
        scale      = Observable( s_0 )
        head_scale = Observable( a_0 ); 
        arrowsize  = Observable( M_0[:] .* a_0 ); 
    end    

    f = GLMakie.Figure(size = (400, 800), backgroundcolor = :gray80, dpi=150 )

    # setting up the top plot: image data + vector fields
    begin top = f[1,1] = GridLayout(); 

        a1 = GLMakie.Axis( top[1,1], backgroundcolor = "black", aspect=DataAspect() )
        hidedecorations!( a1, grid = false )
        Makie.image!( a1, I, colormap=cmap_data )
        if isnothing( arrow_color )
            Makie.arrows!( a1, ygrid_sub, xgrid_sub, U, V, arrowsize = arrowsize, lengthscale = scale, arrowcolor = M, linecolor = M )
        else
            Makie.arrows!( a1, ygrid_sub, xgrid_sub, U, V, arrowsize = arrowsize, lengthscale = scale, arrowcolor = arrow_color, linecolor = arrow_color )
        end
    end

    # setting up parameter widgets at the bottom
    begin bot = f[2,1] = GridLayout(); 

        time_panel = bot[1,1] = GridLayout(); 

        t_lbl, t_tbox, t_prev, t_slid, t_next = make_time_slider( time_panel, tp0=1, tp1=length(PIVs) )
    
        scale_panel = bot[2,1] = GridLayout(); 
    
        scale_lbl, scale_tbox, smin_tbox, scale_slid, smax_tbox = make_scale_slider( scale_panel, scale_range=scale_range )
    
        head_panel = bot[3,1] = GridLayout(); 
    
        head_lbl, head_tbox, hmin_tbox, head_slid, hmax_tbox = make_scale_slider( head_panel, title="arrow heads: ", scale_range=arrow_range, scale_0=a_0 )   

        # time control logic

        on(t_slid.value) do tp
            U_ = PIVs[tp][1,vf_yrange,vf_xrange]
            V_ = PIVs[tp][2,vf_yrange,vf_xrange]
            M_ = sqrt.( U_.^2 .+ V_.^2 )
            M2 = copy( M_ )
            M2[ M2 .> max_speed ] .= max_speed
            U[] = U_ ./ M_ .* M2
            V[] = V_ ./ M_ .* M2
            I[] = images[tp]
            M[] = M2[:]
            arrowsize[] = M2[:] .*  head_scale[]
            Makie.set!( t_tbox, string( tp ) )
        end

        # arrow scale control logic

        on(scale_tbox.stored_string) do s
            scale[] = parse(Float64, (s == "") ? string( s_0 ) : s)
            if parse( Float64, s ) != scale_slid.value[]
                set_close_to!( scale_slid, parse( Float64, s ) )
            end
        end

        # arrow head control logic

        on(head_tbox.stored_string) do s
            head_scale[] =  parse(Float64, (s == "") ? string( 1.0 ) : s)
            arrowsize[] = M[] .* head_scale[]
            if parse( Float64, s ) != head_slid.value[]
                set_close_to!( head_slid, parse( Float64, s ) )
            end
        end
    end

    # Sizing of the elements

    colsize!( f.layout, 1, Relative(1) )
    colsize!( bot, 1, Relative(1) )

    display( f )
end

"""
    Specialized image for showing images and 1 overlays (such as wave segmentation or divergence)
"""
function scrollable_vectorfield( PIVs, images, overlays; 
                                 IA=(16,16), step=(16,16), vf_subsample=(1,1),
                                 fig_width=400,
                                 fig_height=800,
                                 figsize=(fig_width,fig_height), 
                                 scale_range=0.0:0.1:2.0, 
                                 arrow_range=0.0:0.1:1.0, 
                                 max_speed=Inf,
                                 min_speed=0.0, 
                                 alpha=0.4, 
                                 cmap_data=:grays,
                                 cmap_overlay=:viridis, 
                                 arrow_color=nothing,
                                 dpi=200,
                                 px_per_unit=1 )
    # vector fields coordinates
    xgrid     = [ ( x - 1 )*step[2] + div(IA[2],2) for x in 1:size(PIVs[1],3) ]
    ygrid     = [ ( y - 1 )*step[1] + div(IA[1],2) for y in 1:size(PIVs[1],2) ]
    vf_yrange = 1:vf_subsample[1]:size(PIVs[1],2)
    vf_xrange = 1:vf_subsample[1]:size(PIVs[1],3)
    xgrid_sub = [ ( x - 1 )*step[2] + div(IA[2],2) for x in vf_xrange ]
    ygrid_sub = [ ( y - 1 )*step[1] + div(IA[1],2) for y in vf_yrange ]

    GLMakie.closeall()

    begin # Observables

        # Initial value of observables, aka the first timepoint
        U_0 = PIVs[1][1,vf_yrange,vf_xrange]
        V_0 = PIVs[1][2,vf_yrange,vf_xrange]
        M_0 = sqrt.( U_0 .^ 2 .+ V_0 .^ 2 );
        s_0 = (scale_range[end]+scale_range[1])/2
        a_0 = (arrow_range[end]+arrow_range[1])/2

        # Observables
        M = Observable( M_0[:] )
        U = Observable( U_0 ./ M_0 .* M_0  )
        V = Observable( V_0 ./ M_0 .* M_0  )
        I = Observable( images[1] ); 
        O = Observable( overlays[1] );
        scale      = Observable( s_0 )
        head_scale = Observable( a_0 ); 
        arrowsize  = Observable( M_0[:] .* a_0 ); 
    end    

    f = GLMakie.Figure(size = (400, 800), backgroundcolor = :gray80, dpi=150 )

    # setting up the top plot: image data + vector fields
    begin top = f[1,1] = GridLayout(); 

        a1 = GLMakie.Axis( top[1,1], backgroundcolor = "black", aspect=DataAspect() )
        hidedecorations!( a1, grid = false )
        Makie.image!( a1, I, colormap=cmap_data )
        Makie.image!( a1, O, alpha=alpha, colormap=cmap_overlay )
        if isnothing( arrow_color )
            Makie.arrows!( a1, ygrid_sub, xgrid_sub, U, V, arrowsize = arrowsize, lengthscale = scale, arrowcolor = M, linecolor = M )
        else
            Makie.arrows!( a1, ygrid_sub, xgrid_sub, U, V, arrowsize = arrowsize, lengthscale = scale, arrowcolor = arrow_color, linecolor = arrow_color )
        end
    end

    # setting up parameter widgets at the bottom
    begin bot = f[2,1] = GridLayout(); 

        time_panel = bot[1,1] = GridLayout(); 

        t_lbl, t_tbox, t_prev, t_slid, t_next = make_time_slider( time_panel, tp0=1, tp1=length(PIVs) )
    
        scale_panel = bot[2,1] = GridLayout(); 
    
        scale_lbl, scale_tbox, smin_tbox, scale_slid, smax_tbox = make_scale_slider( scale_panel, scale_range=scale_range )
    
        head_panel = bot[3,1] = GridLayout(); 
    
        head_lbl, head_tbox, hmin_tbox, head_slid, hmax_tbox = make_scale_slider( head_panel, title="arrow heads: ", scale_range=arrow_range, scale_0=a_0 )   

        save_panel = bot[4,1] = GridLayout(); 

        save_lbl, save_tbox, save_button = make_save_option( save_panel, target=a1, px_per_unit=px_per_unit ); 

        # time control logic

        on(t_slid.value) do tp
            U_ = PIVs[tp][1,vf_yrange,vf_xrange]
            V_ = PIVs[tp][2,vf_yrange,vf_xrange]
            M_ = sqrt.( U_.^2 .+ V_.^2 )
            M2 = copy( M_ )
            M2[ M2 .> max_speed ] .= max_speed
            U[] = U_ ./ M_ .* M2
            V[] = V_ ./ M_ .* M2
            I[] = images[tp]
            O[] = overlays[tp]
            M[] = M2[:]
            arrowsize[] = M2[:] .*  head_scale[]
            Makie.set!( t_tbox, string( tp ) )
        end

        # arrow scale control logic

        on(scale_tbox.stored_string) do s
            scale[] = parse(Float64, (s == "") ? string( s_0 ) : s)
            if parse( Float64, s ) != scale_slid.value[]
                set_close_to!( scale_slid, parse( Float64, s ) )
            end
        end

        # arrow head control logic

        on(head_tbox.stored_string) do s
            head_scale[] =  parse(Float64, (s == "") ? string( 1.0 ) : s)
            arrowsize[] = M[] .* head_scale[]
            if parse( Float64, s ) != head_slid.value[]
                set_close_to!( head_slid, parse( Float64, s ) )
            end
        end
    end

    # Sizing of the elements

    colsize!( f.layout, 1, Relative(1) )
    colsize!( bot, 1, Relative(1) )

    display( f )
end

function make_time_slider( parent; title="TP: ", tp0=1, tp1=100 )

    Box( parent[1, 1:5], color = (:white, 0.0), strokewidth = 0 )
    t_lbl  =   Label( parent[1,1], title )
    t_tbox = Textbox( parent[1,2], placeholder=string(tp0) )
    t_prev =  Button( parent[1,3], label="-1" )
    t_slid =  Slider( parent[1,4], range = 1:tp1, startvalue = tp0 )
    t_next =  Button( parent[1,5], label="+1" ) 

    # Logic between the elements of the slider

    on( t_tbox.stored_string ) do tp
        if parse( Int, tp ) != t_slid.value[]
            set_close_to!( t_slid, parse( Int, tp ) )
        end
    end
    
    on(t_prev.clicks) do n
        set_close_to!( t_slid, t_slid.value[] - 1 )
    end
    on(t_next.clicks) do n
        set_close_to!( t_slid, t_slid.value[] + 1 )
    end

    return t_lbl, t_tbox, t_prev, t_slid, t_next
end

function make_scale_slider( parent; title="vector scale: ", s0=1, s1=100, step=1, scale_range=s0:step:s1, scale_0=(scale_range[end]+scale_range[1])/2  )

    Box( parent[1, 1:5], color = (:white, 0.0), strokewidth = 0 )
    scale_lbl  =   Label( parent[1,1], title )
    scale_tbox = Textbox( parent[1,2], placeholder=string(scale_0) )
    smin_tbox  = Textbox( parent[1,3], placeholder=string(scale_range[1]) )
    scale_slid =  Slider( parent[1,4], range = scale_range, startvalue = scale_0 )
    smax_tbox  = Textbox( parent[1,5], placeholder=string(scale_range[end]) )

    # Logic between the elements of the slider

    on(smin_tbox.stored_string) do s
        r0 = parse(Float64,s)
        r1 = parse(Float64,smax_tbox.displayed_string[])
        st = (r1-r0)/99
        scale_slid.range = r0:st:r1
    end
    
    on(smax_tbox.stored_string) do s
        r0 = parse(Float64,smin_tbox.displayed_string[])
        r1 = parse(Float64,s)
        st = (r1-r0)/99
        scale_slid.range = r0:st:r1
    end
    
    on(scale_slid.value) do s
        Makie.set!( scale_tbox, string( s ) )
    end

    return scale_lbl, scale_tbox, smin_tbox, scale_slid, smax_tbox
end

function make_save_option( parent; title="output file: ", default_output="R:/users/pereyram/TMP_BACKUP/screenshot.png", target=nothing, px_per_unit=1 )

    Box( parent[1, 1:3], color = (:white, 0.0), strokewidth = 0 )
    t_lbl  =   Label( parent[1,1], title )
    t_tbox = Textbox( parent[1,2], placeholder=default_output, stored_string=default_output )
    t_save =  Button( parent[1,3], label="Save" ) 

    on(t_save.clicks) do n
        output_ = t_tbox.displayed_string[]; 
        save(output_, target.scene, px_per_unit=px_per_unit )
    end

    return t_lbl, t_tbox, t_save
end

