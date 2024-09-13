function post_process_waves( WVs; min_size=10, max_height=20 )

    tmp = remove_small_V_phases( WVs, min_size^2 )
    tmp = remove_small_D_phases( tmp, min_size^2 )
    tmp = opening_V_phases( tmp )
    tmp = opening_D_phases( tmp )
    tmp = remove_small_V_phases( tmp, min_size^2 )
    tmp = remove_small_D_phases( tmp, min_size^2 )
    tmp = remove_high_V_phases( tmp, max_height ); 
    tmp = remove_high_D_phases( tmp, max_height ); 
    tmp = remove_orphan_V_phases( tmp )
    filt_WVs = remove_orphan_D_phases( tmp );

    return filt_WVs
end

function post_process_cylinder_waves( WVs; min_size=15, max_width=40 )

    tmp = remove_small_V_phases( WVs, min_size^2 )
    tmp = remove_small_D_phases( tmp, min_size^2 )
    tmp = opening_V_phases( tmp )
    tmp = opening_D_phases( tmp )
    tmp = remove_small_V_phases( tmp, min_size^2 )
    tmp = remove_small_D_phases( tmp, min_size^2 )
    tmp = remove_right_V_phases( tmp, max_width ); 
    tmp = remove_right_D_phases( tmp, max_width ); 
    tmp = remove_orphan_V_phases( tmp )
    filt_WVs = remove_orphan_D_phases( tmp );

    return filt_WVs
end

function smooth_waves( WVs; rad=3, th=0.5 )

    ksize = ones( Int, ndims(WVs) ) .* ( 2 * rad + 1 ) 
    kern  = ones( Float32, ksize... ) ./ prod( ksize )
    smth_WVs   = multi_quickPIV.FFTCC_crop( kern, Float32.( WVs ) )
    smth_WVs .*= abs.( smth_WVs ) .> th
    smth_WVs[ abs.( smth_WVs ) .> 0 ] ./= abs.( smth_WVs[ abs.( smth_WVs ) .> 0 ] )
    return smth_WVs
end

function kymographs( waves::Vector{Array{T,2}}; t0=1, t1=length(waves), tmin=1, tmax=length(waves), fun=(x)->sum(x) ) where {T}
    
    t0 = max(t0,tmin)
    t1 = min(t1,tmax)
    
    Ntps = tmax - tmin + 1; 
    h, w = size( waves[1] ); 
    
    kymo_1 = zeros( Float32, h, Ntps )
    kymo_2 = zeros( Float32, Ntps, w )
    
    for t in t0:t1
        
        # takes each row and applies fun to it ( usually sum() or maximum() )
        for row in 1:h
            kymo_1[ row, t ] = fun( waves[t][row,:] )
        end
        
        # takes each column and applies fun to it ( usually sum() or maximum() )
        for col in 1:w
            kymo_2[ t, col ] = fun( waves[t][:,col] )
        end
    end
    
    return kymo_1, kymo_2    
end

function kymographs( waves::Vector{Any}; t0=1, t1=length(waves), tmin=1, tmax=length(waves), fun=(x)->sum(x) )
    
    t0 = max(t0,tmin)
    t1 = min(t1,tmax)
    
    Ntps = tmax - tmin + 1; 
    h, w = size( waves[1] ); 
    
    kymo_1 = zeros( Float32, h, Ntps )
    kymo_2 = zeros( Float32, Ntps, w )
    
    for t in t0:t1
        
        # takes each row and applies fun to it ( usually sum() or maximum() )
        for row in 1:h
            kymo_1[ row, t ] = fun( waves[t][row,:] )
        end
        
        # takes each column and applies fun to it ( usually sum() or maximum() )
        for col in 1:w
            kymo_2[ t, col ] = fun( waves[t][:,col] )
        end
    end
    
    return kymo_1, kymo_2    
end

remove_small_V_phases( waves, min_area ) = remove_small_phases( waves, min_area, 1 )
remove_small_D_phases( waves, min_area ) = remove_small_phases( waves, min_area, -1 )

function remove_small_phases( waves, min_area, val=1 )

    Ntps  = length( waves ); 
    wsize = size( waves[1] );
    ND    = length( wsize ); 

    V = zeros( UInt8, wsize );

    connectivity = zeros( Bool, ( ones( Int, ND ) .*  3 )... ) 
    for ci in CartesianIndices(connectivity)
        connectivity[ci] = sum( abs.(Tuple(ci) .- ones(Int,ND).*2) ) < 2 
    end

    output = []

    for wi in 1:Ntps

        wv = deepcopy( waves[ wi ] ); 
        V .= UInt8.( wv .== val  ); 

        labels = ImageComponentAnalysis.label_components( V, connectivity ); 
        N_lbls = maximum( labels );
        areas  = zeros( Int, N_lbls + 1 ); 

        for i in 1:length( labels )
            lbl = labels[i]
            areas[ lbl + 1 ] += 1 
        end

        for n in 2:N_lbls+1
            if areas[n] < min_area
                wv[ labels .== ( n - 1 ) ] .= 0
            end
        end
        
        push!( output, wv )
    end

    return output
end

remove_high_V_phases( waves, min_y ) = remove_high_phases( waves, min_y, 1 )
remove_high_D_phases( waves, min_y ) = remove_high_phases( waves, min_y, -1 )

function remove_high_phases( waves, min_y, val=1 )

    Ntps  = length( waves ); 
    wsize = size( waves[1] )
    ND    = length( wsize )
    V     = zeros( UInt8, wsize );
    connectivity = zeros( Bool, ( ones( Int, ND ) .*  3 )... ) 
    for ci in CartesianIndices(connectivity)
        connectivity[ci] = sum( abs.(Tuple(ci) .- ones(Int,ND).*2) ) < 2 
    end

    output = []

    for wi in 1:Ntps

        wv = deepcopy( waves[ wi ] ); 
        V .= UInt8.( wv .== val  ); 

        labels = ImageComponentAnalysis.label_components( V, connectivity ); 
        N_lbls = maximum( labels );
        y_sum  = zeros( Int, N_lbls ); 
        N_sum  = zeros( Int, N_lbls ); 

        for ci in CartesianIndices( labels )
            lbl = labels[ ci ]
            if lbl == 0 
                continue
            end
            y_sum[ lbl ] += ci[1]
            N_sum[ lbl ] += 1
        end

        for n in 1:N_lbls
            y_centroid = y_sum[n]/N_sum[n]
            if y_centroid < min_y
                wv[ labels .== n ] .= 0
            end
        end
        
        push!( output, wv )
    end

    return output
end

remove_right_V_phases( waves, min_x ) = remove_right_phases( waves, min_x, 1 )
remove_right_D_phases( waves, min_x ) = remove_right_phases( waves, min_x, -1 )

function remove_right_phases( waves, min_x, val )

    Ntps  = length( waves ); 
    wsize = size( waves[1] )
    ND    = length( wsize )
    V     = zeros( UInt8, wsize );
    connectivity = zeros( Bool, ( ones( Int, ND ) .*  3 )... ) 
    for ci in CartesianIndices(connectivity)
        connectivity[ci] = sum( abs.(Tuple(ci) .- ones(Int,ND).*2) ) < 2 
    end

    output = []

    for wi in 1:Ntps

        wv = deepcopy( waves[ wi ] ); 
        V .= UInt8.( wv .== val  ); 

        labels = ImageComponentAnalysis.label_components( V, connectivity ); 
        N_lbls = maximum( labels );
        x_sum  = zeros( Int, N_lbls ); 
        N_sum  = zeros( Int, N_lbls ); 

        for ci in CartesianIndices( labels )
            lbl = labels[ ci ]
            if lbl == 0 
                continue
            end
            x_sum[ lbl ] += ci[2]
            N_sum[ lbl ] += 1
        end

        for n in 1:N_lbls
            x_centroid = x_sum[n]/N_sum[n]
            if x_centroid > min_x
                wv[ labels .== n ] .= 0
            end
        end
        
        push!( output, wv )
    end

    return output
end

function opening_phases( waves, val=1 )

    Ntps   = length( waves ); 
    w_size = size( waves[1] ); 
    V      = zeros( UInt8, w_size ); 
    wv_tmp = zeros( eltype(waves[1]), w_size ); 
    output = []

    for t in 1:Ntps
        wv_tmp .= waves[t]
        V      .= UInt8.( wv_tmp .== val ); 
        V_open  = ImageMorphology.opening( V ); 
        wv_tmp[ V .== val ] .*= V_open[ V .== val ]; 
        push!( output, deepcopy( wv_tmp ) )
    end
    return output
end

opening_V_phases( waves ) = opening_phases( waves, 1 )
opening_D_phases( waves ) = opening_phases( waves, -1 )

function fill_new_V_phases( waves )
    # 1-. compute avg duration of each V phase
    # 2-. If a V phase isn't followed by a D phase, add a D phase of average duration
end

function remove_orphan_D_phases( waves )

    N_tps  = length( waves ); 
    w_size = size( waves[1] ); 
    w_len  = prod( w_size );

    last_v = zeros( Int, w_size )
    last_d = zeros( Int, w_size )
    w_tmp  = zeros( eltype( waves[1] ), w_size ); 

    out = []

    for t in 1:N_tps 

        w_tmp .= waves[t]
        
        for i in 1:w_len

            is_v = ( w_tmp[i] == 1 )
            if is_v
                last_v[i] = t
                continue
            end

            is_d = ( w_tmp[i] == -1 )
            d_before_v = ( last_d[i] > last_v[i] )
            last_d_isnt_adjacent = ( ( t - last_d[i] ) > 1 )

            if is_d
                if ( last_v[i] == 0 ) || ( d_before_v && last_d_isnt_adjacent )
                    w_tmp[i] = 0
                else
                    last_d[i] = t
                end
            end
        end
        push!( out, deepcopy( w_tmp ) )
    end

    return out
end

function remove_orphan_V_phases( waves )

    N_tps  = length( waves ); 
    w_size = size( waves[1] ); 
    w_len  = prod( w_size );

    last_v = zeros( Int, w_size )
    last_d = zeros( Int, w_size )
    w_tmp  = zeros( eltype( waves[1] ), w_size ); 

    out = []

    for t in 1:N_tps 

        w_tmp .= waves[t]
        
        for i in 1:w_len

            is_d = ( w_tmp[i] == -1 )
            if is_d
                last_d[i] = t
                continue
            end

            is_v = ( w_tmp[i] == 1 )
            v_before_d = ( last_v[i] > last_d[i] )
            last_v_isnt_adjacent = ( ( t- last_v[i] ) > 1 )

            if is_d
                if ( last_d[i] == 0 ) || ( v_before_d && last_v_isnt_adjacent )
                    w_tmp[i] = 0
                else
                    last_v[i] = t
                end
            end
        end
        push!( out, deepcopy( w_tmp ) )
    end

    return out
end

function remove_hanging_segmentations( waves, max_distance )
    waves_ = deepcopy( waves ); 
    Ntps   = length( waves )
    for c in CartesianIndices( size( waves_[1] ) )
        for t in 1+1:Ntps-1
            if waves[t][c] == 1 && waves[t+1][c] == 0
                is_not_hanging = false; 
                for tt in t+1:min(Ntps,t+max_distance)
                    is_not_hanging = is_not_hanging || waves[tt][c] == -1
                end
                if !is_not_hanging
                    tt = t
                    while ( waves[tt][c] == 1 ) && ( tt >= 1 )
                        waves_[tt][c] = 0
                        tt -= 1
                    end
                end
            end
            if waves[t][c] == -1 && waves[t-1][c] == 0
                is_not_hanging = false; 
                for tt in t-1:-1:max(1,t-max_distance)
                    is_not_hanging = is_not_hanging || waves[tt][c] == 1
                end
                if !is_not_hanging
                    tt = t
                    while ( waves[tt][c] == -1 ) && ( tt < Ntps )
                        waves_[tt][c] = 0
                        tt += 1
                    end
                end
            end
        end
    end
    return waves_
end

function fill_hanging_segmentations( waves, max_distance )
    waves_ = deepcopy( waves ); 
    Ntps   = length( waves )
    for c in CartesianIndices( size( waves_[1] ) )
        for t in 1+2:Ntps-2
            if waves[t][c] == 1 && waves[t+1][c] == 0
                is_not_hanging = false; 
                D_start = 0
                for tt in t+1:min(Ntps,t+max_distance)
                    is_hanging = !is_not_hanging
                    D_start = is_hanging ? ( waves[tt][c] == -1 ) : D_start; 
                    is_not_hanging = is_not_hanging || waves[tt][c] == -1
                end
                if is_not_hanging
                    len = D_start - (t+1) + 1
                    half = div( len, 2 )
                    for tt in t+1:t+half
                        waves_[tt][c] = 1
                    end
                    for tt in t+half+1:t+len
                        waves_[tt][c] = -1
                    end
                end
            end
        end
    end
    return waves_
end