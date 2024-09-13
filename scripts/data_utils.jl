include("segment_utils.jl")

begin # LOADING DATA 
    # 10 -> 0010
    function replace_lpad( target, regx, value )
        
        match_ = match( regx, target ); 
        match_str = match_.match
        
        capture_   = match_[1]
        lpad_value = lpad( string(value), length( capture_ ), '0' )
        
        formatted_value = replace( match_str, capture_ => lpad_value )
        
        return replace( target, match_str => formatted_value )
    end

    # 10 -> 10
    function replace_nolpad( target, regx, value )
        
        match_ = match( regx, target ); 
        match_str = match_.match
        
        capture_   = match_[1]
        lpad_value = string( value )
        
        formatted_value = replace( match_str, capture_ => lpad_value )
        
        return replace( target, match_str => formatted_value )
    end

    function replace_tp( target, regx, value; add_pad=false )
        if add_pad
            return repalce_lpad( target, regx, value )
        else
            return replace_nolpad( target, regx, value )
        end
    end

    # figure out data dimensions, so that we can preallocate an array with the data's type and size for in-place tiffread!
    function tiff_type_and_size( filename::String; subsample=ones(Int,3) )

        f = LIBTIFF.tiffopen(filename)

        N = length( f.dims ); 
        axes = StepRange.( 1, subsample[1:N], f.dims )
        
        T = eltype(f); 

        LIBTIFF.tiffclose(f)
        return T, length.( axes )
    end
end

begin # CREATING LATERAL MAX PROJECTION FROM 3D+t FUSED VOLUMES

    # maxprojection
    function maxprojection!( output::AbstractArray{T,M}, 
                            data::AbstractArray{T,N}; 
                            axis=N, # axis along which to project the data
                            ROI=axes(data), # A subregion of the data can be projected. Defaults to the whole input data.
                            fun=(x)->(maximum(x)) # projection function. Defaults to max projection
                            ) where {T,N,M}

        # creating an index of the form [ 1, 1, : ] or [ 1, :, 1 ], where the projection "axis" contains a range over all of its selements
        data_index = [ ( n == axis ) ? ROI[n] : 1 for n in 1:N ]; 

        # Selecting all axes in the data, except the projection "axis". 
        axes_mask  = ( collect(1:N) .!= axis ); 
        proj_axes  = Tuple( ROI[ axes_mask ] ); 
        proj_size  = length.( proj_axes )

        # populating the projection
        for proj_index in CartesianIndices( proj_axes )
            data_index[ axes_mask  ] .= Tuple( proj_index )
            output[ proj_index ]  = fun( view( data, data_index... ) )
        end
        
        return nothing
    end

    # maxprojection
    function maxprojection( data::AbstractArray{T,N}; axis=N, ROI=axes(data), fun=(x)->(maximum(x)) ) where {T,N}

        # creating an index of the form [ 1, 1, : ] or [ 1, :, 1 ], where the projection "axis" contains a range over all of its selements
        data_index = [ ( n == axis ) ? ROI[n] : 1 for n in 1:N ]; 

        # Selecting all axes in the data, expect the projection "axis". 
        axes_mask  = ( collect(1:N) .!= axis ); 
        proj_axes  = Tuple( ROI[ axes_mask ] ); 
        proj_size  = length.( proj_axes )
        projection = zeros( T, proj_size... );

        maxprojection!( projection, data, axis=axis, ROI=ROI, fun=fun )
        
        return projection
    end

    function generate_lateral_projections( volume_path, tp_regex, tp0 ,tp1 )

        N_tps = tp1 - tp0 + 1;
        vol_type, vol_size = tiff_type_and_size( replace_tp( volume_path, tp_regex, tp0 ) )

        tmp_volume  = zeros( vol_type, vol_size... ); 
        tmp_maxproj = zeros( vol_type, vol_size[1], vol_size[3] ); 
        lateral_2Dt = zeros( vol_type, vol_size[1], vol_size[3], N_tps ); 

        mp_axis = 2;
        h, w, d = vol_size;
        ROI_dr4 = ( 1:h,  1:div(w,2) , 1:d );

        for t in tp0:tp1
            
            print( t, "," )
            tp_path = replace_tp( volume_path, tp_regex, t ); 
            LIBTIFF.tiffread!( tmp_volume, tp_path ); 
            
            maxprojection!( tmp_maxproj, tmp_volume, axis=mp_axis, ROI=ROI_dr4 ); 
            
            lateral_2Dt[:,:,t-tp0+1] .= tmp_maxproj; 
        end

        return lateral_2Dt
    end
end

begin # COMPUTING A MASK FROM MULTIPLE LATERAL PROJECTIONS FOR MASKED PIV

    function get_embryo_mask( lateral_2Dt, mask_Ntps=10 )

        grid_sizes = ( (7,7), )
        scales     = ( ((4,4), (1,1)), )
        steps      = ( ((1,1), (3,3)), )
        min_dif    = 10
        alpha      = 1.0

        mask = zeros( UInt8, size( lateral_2Dt )[1:2] ); 
        Ntps = size( lateral_2Dt, 3 ); 
        for t in 1:div((Ntps-1),mask_Ntps):Ntps
            mask_tmp = multiscale_multistep_segment( lateral_2Dt[:,:,t], grid_sizes = grid_sizes, scales = scales, steps = steps, min_dif = min_dif, alpha = alpha );
            clean_mask!( mask_tmp, remove_small=true, min_size=2^2, remove_borders=true, min_border_distance=(10,10,0) );
            mask .= max.( mask, paint_chull_interior( mask_tmp ) );
        end

        return mask
    end
end