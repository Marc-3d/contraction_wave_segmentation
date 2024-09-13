function post_process_laterals_VFs( VFs; th1=Inf, th2=0.6 )
    postVFs = []
    for VF in VFs
        VFmask_1 = local_magnitude_mask( th1, (4,4), VF )
        VFmask_2 = collectiveness_mask( th2, (2,2), VF ) 
        VFmask   = max.( VFmask_1, VFmask_2 ) .* ( magnitudes( VF ) .> 0.1 )
        VF_      = average_masked_vectors!( VFmask, copy(VF), radii=(2,2), niters=8 )
        push!( postVFs, VF_ )
    end

    return postVFs
end

begin # utils

    """
        vfsize( U, V[, W] ) or vfsize( VF )
    """
    vfsize( inp::Vararg{AbstractArray} ) = vfsize( inp )

    # vfsize( (VF,) )
    vfsize( inp::Tuple{AbstractArray} ) = size( inp[1] )[2:end]

    # vfsize( (U,V[,W]) )
    vfsize( inp::NTuple{N,AbstractArray} ) where {N} = size( inp[1] )

    """
        num_components( U, V[, W] ) or num_components( VF )
    """
    num_components( inp::Vararg{AbstractArray} ) = num_components( inp )

    # num_components( (VF,) )
    num_components( inp::Tuple{AbstractArray} ) = size( inp[1], 1 )

    # num_components( (U,V[,W]) )
    num_components( inp::NTuple{N,AbstractArray} ) where {N} = N

    """
        get_vec( (y,x[,z]), U, V[, W] ) or get_vec( (y,x[,z]), VF )
    """
    get_vec( coord, inp::Vararg{AbstractArray} ) = get_vec( coord, inp )

    # get_vec( (y,x[,z]), (VF,) )
    get_vec( coord, inp::Tuple{AbstractArray}  ) = inp[1][:,coord...]

    # get_vec( (y,x[,z]), (U,V[,W]) )
    get_vec( coord, inp::NTuple{N,AbstractArray} ) where {N} = [ inp[c][coord...] for c in 1:N ]  

    """
        set_val!( newval, (y,x[,z]), U, V[, W] ) or num_components( newval, (y,x[,z]), VF )
    """
    function set_val!( newval, coord, inp::Vararg{AbstractArray} )
        set_val!( newval, coord, inp )
        return nothing
    end

    # set_val!( (u,v[,w]), (y,x[,z]), (VF,) )
    set_val!( newval, coord, inp::Tuple{AbstractArray} ) = inp[1][:,coord...] .= newval

    # set_val!( (u,v[,w]), (y,x[,z]), (U,V[,W]) )
    set_val!( newval, coord, inp::NTuple{N,AbstractArray} ) where {N} = [ inp[c][coord...] .= newval[c] for c in 1:N ]

    # set_val!( value, (y,x[,z]), (data,) )
    set_val!( newval::Number, coord::NTuple{N,Int}, inp::Tuple{AbstractArray{T,N}}  ) where {T,N} = inp[1][coord...] = newval

    """

    """
    multiply!( weights, inp::Vararg{AbstractArray} ) = multiply!( weights, inp )

    # multiply!( weights, (VF,) )
    function multiply!( weights, inp::Tuple{AbstractArray} )
        [ inp[1][ c, UnitRange.( 1, vfsize(inp) )... ] .*= weights for c in 1:num_components( inp ) ] 
        return nothing
    end

    # multiply!( weights, (U,V,W) )
    function multiply!( weights, inp::NTuple{N,AbstractArray} ) where {N}
        [ inp[c] .*= weights for c in 1:N ]
        return nothing
    end

    """

    """
    divide!( weights, inp::Vararg{AbstractArray} ) = divide!( weights, inp )

    # divide!( weights, (VF,) )
    function divide!( weights, inp::Tuple{AbstractArray} )
        [ inp[1][ c, UnitRange.( 1, vfsize(inp) )... ] ./= weights for c in 1:num_components( inp ) ] 
        return nothing
    end

    # divide!( weights, (U,V,W) )
    function divide!( weights, inp::NTuple{N,AbstractArray} ) where {N}
        [ inp[c] ./= weights for c in 1:N ]
        return nothing
    end

    """
        local_op_dimwise( op, (y0:y1,x0:x1,z0:z1), U, V, W ) or local_op_dimwise( op, (20:40,20:40,3:8), VF )
    """
    local_op_dimwise( op, ROI, inp::Vararg{AbstractArray} ) = local_op_dimwise( op, ROI, inp )

    # local_op_dimwise( op, ROI, (VF,) )
    local_op_dimwise( op, ROI, inp::Tuple{AbstractArray} ) = [ op( inp[1][ c, ROI... ] ) for c in 1:num_components( inp ) ] 

    # local_op_dimwise( op, ROI, (U,V,W) )
    local_op_dimwise( op, ROI, inp::NTuple{N,AbstractArray} ) where {N} = [ output[c] = op( inp[c][ ROI... ] ) for c in 1:N ]

    """
        This function applies a function within a region of interest in of the input array. This is used
        all the time to compute local quantities withing local ROI around each element in the input.
    """
    local_op( op, ROI, inp::AbstractArray ) = op( inp[ ROI... ] )

    """
        Returns the indices to access the neighbouring regions around "coords"
    """
    localROI( coords, radii, vf_size ) = UnitRange.( max.( 1, coords .- radii ), min.( vf_size, coords .+ radii ) )

    """
        Locally applies the same function to all components of a vector field at each position, 
        and further applied a function to the previous output. 
    """
    function local_mapreduce( radii::NTuple{N,Int}, 
                              vectorfield::AbstractArray...; 
                              map=(x)->( sum(x)/length(x) ),
                              mapreduce=(x)->(sum(x)),
                              out_type = Float32, 
                              skip_assert = false
                            ) where {N}

        vf_dims   = num_components( vectorfield )
        @assert skip_assert || N == vf_dims

        vf_size  = vfsize( vectorfield )
        mpr_size = length( mapreduce( local_op_dimwise( map, localROI( div.(vf_size,2), radii, vf_size ), vectorfield ) ) ); 
        out_size = ( mpr_size > 1 ) ? ( mpr_size, vf_size... ) : vf_size;
        output   = zeros( out_type, out_size )

        for c in CartesianIndices( vf_size )
            ROI  = localROI( Tuple( c ), radii, vf_size );
            map_ = local_op_dimwise( map, ROI, vectorfield );
            out_ = mapreduce( map_ ); 
            set_val!( out_, Tuple(c), output )
        end

        return output
    end

    """
        This function is required to compute the local average of a single array, not a vector field.
    """
    function local_map( radii::NTuple{N,Int}, 
                        inp::AbstractArray; 
                        map=(x)->( sum(x)/length(x) ),
                        out_type = Float32, 
                      ) where {N}

        output = zeros( out_type, size(inp) )

        for c in CartesianIndices( size(inp) )
            ROI  = localROI( Tuple( c ), radii, size(inp) );
            out_ = local_op( map, ROI, inp );
            set_val!( out_, Tuple(c), output )
        end

        return output
    end
end

begin # postprocessing.jl

    """
        Usually used for spatial smoothing.

        If multiple vector fields are combined into one array, this function can also be used for temporal and
        spatio-temporal smooting.
    """
    function smooth( radii, VF )
        smoothed = local_mapreduce( radii, VF, mapreduce=(x)->(x), skip_assert=true )
        return smoothed
    end

    """
        magnitudes expressed as mapreduce on the vector field
    """
    function magnitudes( VF )
        ndims  = num_components( VF )
        radii  = Tuple(zeros(Int,ndims)); 
        output = local_mapreduce( radii, VF, map=(x)->(x[1]), mapreduce=(x)->(sqrt(sum(x.^2))) )
        return output
    end

    """
        local magnitudes expressed as a local_map on "magnitudes"s
    """
    function local_magnitudes( radii, VF )
        M  = magnitudes( VF ); 
        Ml = local_map( radii, M, map=(x)->((sum(x)-x[div.(size(x).+1,2)...])/(length(x)-1)) )
        return Ml
    end

    """
        magnitude / local_magnitude ratio
    """
    function local_magnitude_ratio( radii, VF )
        M  = magnitudes( VF ); 
        Ml = local_map( radii, M, map=(x)->((sum(x)-x[div.(size(x).+1,2)...])/(sum(x .> 0)-(x[div.(size(x).+1,2)...]>0))) )
        return M ./ Ml
    end

    """
    """
    function local_magnitude_mask( max_ratio, radii, VF )
        mag_ratios = local_magnitude_ratio( radii, VF ); 
        return mag_ratios .>= max_ratio
    end

    """
        normalize vectorfield
    """
    function normalize( VF )
        M = magnitudes( VF )
        output = copy( VF ); 
        divide!( M, output ); 
        return output
    end

    """
        collectiveness
    """
    function collectiveness( radii, VF; max_angle=20, min_speed=0.0 )

        VFn     = normalize( VF ); 
        mags    = magnitudes( VF );
        vf_size = vfsize( VFn );
        output  = zeros( vf_size ); 
        min_dot = cosd( max_angle );
        for c in CartesianIndices( vf_size )
            if mags[c] < min_speed
                continue
            end
            nvec = get_vec( Tuple(c), VFn )
            ROI  = localROI( Tuple(c), radii, vf_size )
            N_similar = 0
            N_total  = 0
            for local_index in CartesianIndices( ROI )
                nvec_ = get_vec( Tuple(local_index), VFn )
                dot_  = sum( nvec .* nvec_ )
                N_similar += dot_ > min_dot
                N_total   += mags[ local_index ] > min_speed
            end
            output[c] = N_similar / N_total
        end
        return output
    end

    """
    """
    function collectiveness_mask( min_collectiveness, radii, VF; max_angle=20, min_speed=0.0 )
        coll = collectiveness( radii, VF, max_angle=max_angle, min_speed=0.0 )
        return coll .< min_collectiveness
    end

    """
        We wish to mask away noisy vectors, before replacing them by the local average. This way, 
        noisy vectors don't contribute to the local averaging that replaces the noisy vectors.
    """
    function average_masked_vectors!( mask::AbstractArray{T,N}, VF; radii=ones(Int,N).*2, niters=4 ) where {T,N}
        
        # applying reversed mask on VF, since mask is 1 at elements to be removed, and 0 at elements to keep. 
        multiply!( mask .== 0, VF )
        
        vf_size = vfsize( VF )
        
        # creating a list of coordinates of the removed elements
        removed   = CartesianIndices( mask )[ mask ]
        n_removed = length( removed ); 
        
        M = UInt8.( magnitudes( VF ) .> 0 ); 
        
        # replacing the removed elements... TODO: consider using distance transforms to determine "niters"
        for n in 1:niters
            
            still_removed = []; 
            
            for ci in removed
                
                avgROI = localROI( Tuple(ci), radii, vf_size )
                
                # counting the number of neighbouring vectors around the removed vector
                sum_N = local_op( (x)->(sum(x)), avgROI, M )
                
                # if region around the removed vector is empty (maybe because it was surrounded by other removed vectors), continue
                if sum_N == 0
                    push!( still_removed, ci )      
                # otherwise, replace the removed vector by the local average
                else 
                    sum_vec = local_op_dimwise( (x)->(sum(x)), avgROI, VF )
                    avg_vec = sum_vec ./ sum_N; 
                    set_val!( avg_vec, Tuple( ci ), VF )
                    M[ ci ] = 1
                end
            end
            removed = still_removed
        end
        return VF
    end
end