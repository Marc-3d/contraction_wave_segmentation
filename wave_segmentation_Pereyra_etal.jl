begin # wave parameters 

    #=
        The algorithm uses integral arrays to compute average vectors and average dot products. Thus, instead
        of using vector components directly, we store the "sums of vector components for each dimension and 
        the sum of magnitudes" for all vectors belonging to a group of vectors. In other words, given a group
        of two vectors, (v1,v2), we would store: 

                    ( v1_x + v2_x, v1_y + v2_y, [ v1_z + v2_z ,], M1 + M2 ) 

        where v1 = ( v1_x, v1_y [, v1_z ] ), v2 = ( v2_x, v2_y [, v2_z ] ) and M1 and M2 are the magnitudes
        of v1 and v2, respectively. 

        from these sums, one can instantly find the average vector of any group of vectors by dividing the
        "sums of vector components" by the number of vectors in the group. Indicentally, if the vectors are
        normalized, the sum of magnitudes is equivalent to the number of vectors in the group. This is true
        for the current application of the wave segmentation algorithm, since we analyze normalized vector 
        fields. 
    =#
    mutable struct wave_parameters{N} 

            dorsal_dir::NTuple{N,Real} # reference dorsal direction
           ventral_dir::NTuple{N,Real} # reference ventral direction 
               top_dir::NTuple{N,Real} # reference top direction

           dorsal_sums # the reference directions are expressed as sums (see above)
          ventral_sums # the reference directions are expressed as sums (see above)
              top_sums # the reference directions are expressed as sums (see above)

             min_angle # minimum avg angle between the vectors in V phase and D phase
     max_V_intra_angle # maximum avg angle (deviation) between vectors within V phase
     max_D_intra_angle # maximum avg angle (deviation) between vectors within D phase
         max_ref_angle # maximum avg angle between the vectors in V phase and the reference (ventral) direction
         min_top_angle # minimum avg angle between the vectors the D phase and the top vector

              theta_r # cosd( min_angle )
              theta_V # cosd( max_V_intra_angle )
              theta_D # cosd( max_D_intra_angle )
             theta_VD # cosd( max_ref_angle )
       max_top_cosine

          V_min_speed # minimum speed of the vectors in the V phases
          D_min_speed # minimum speed of the vectors in the D phases
          V_avg_speed # minimum average speed of the v phases
     min_displacement # minimum cummulative displacement in a contraction wave

    end

    dir2sums( dir ) = ( dir..., sqrt(sum(dir .* dir)) ); 

    # default wave_parameters constructor

    function wave_parameters(; 

                   dorsal_dir = ( 0,0, 1), 
                  ventral_dir = ( 0,0,-1), 
                      top_dir = (-1,0, 0),

                  dorsal_sums = dir2sums( dorsal_dir),
                 ventral_sums = dir2sums(ventral_dir),
                     top_sums = dir2sums(  top_dir  ),

                    min_angle = 30, 
            max_V_intra_angle = 50, 
            max_D_intra_angle = 50, 
                max_ref_angle = 40,
                min_top_angle = 60,

                      theta_r = cosd(   max_ref_angle   ),
                      theta_V = cosd( max_V_intra_angle ),
                      theta_D = cosd( max_D_intra_angle ),
                     theta_VD = cosd(     min_angle     ),
               max_top_cosine = cosd(   min_top_angle   ),

                  V_min_speed = 0,
                  D_min_speed = 0,
                  V_avg_speed = 0,
             min_displacement = 5,
            )

    return wave_parameters( dorsal_dir, ventral_dir, top_dir, 
                            dorsal_sums, ventral_sums, top_sums, 
                            min_angle, max_V_intra_angle, max_D_intra_angle,
                            max_ref_angle, min_top_angle,
                            theta_r, theta_V, theta_D, theta_VD, max_top_cosine,
                            V_min_speed, D_min_speed, V_avg_speed, min_displacement
                        )
    end
end

"""
    segment_params = ...
    waves_segmentation = detect_waves( )
"""
function detect_waves( VFs::Vector{<:Any}, 
                       wave_params; 
                       # optional arguments for providing individual reference vectors for each element in the input vector fields.
                       vec_override = false,
                       U_override = nothing,
                       V_override = nothing, 
                       W_override = nothing,
                       # scale of ref_wdot(...), aka how many temporal neighbours to consider in each direction when computing rho(t). 
                       r       =  1,  
                       r_left  =  1, 
                       r_right =  1, 
                       # exclusion radius of local minima and maxima. In other words, if two local minima/maxima are less than "mn/mx_rad" away, the
                       # most extreme one is selected and the other is removed. 
                       mn_rad = 1, 
                       mx_rad = 1,
                       # maximum extent for the V and D phase segmentations
                       r_max = 10, 
                       # 
                       v_min_len=0,
                       d_min_len=0, 
                       #
                       c_idx=1
                     )

    # Number of timepoints == length of the list of vector fields.
    Ntps = length( VFs ); 
    
    # Dimensions of each vector field in the list of vector fields.
    vfsize = VF_size( VFs[1]... ); 
    vfdims = length( vfsize );
    
    # Output segmentation for each vector field in the list of vector fields. (+1 V phase, -1 D phase)
    out = [ zeros( Int8, vfsize ) for t in 1:Ntps ];

    # Computing vector magnitudes and normalizing inputs
    mags_VFs = VF_magnitudes.( VFs );
    norm_VFs = normalize_VF.( VFs );
    
    # For each spatial coordinate...
    for cartesian_pos in CartesianIndices( vfsize );

        # TOOD: generalize this. 
        # Updating ventral/dorsal vectors for cartography
        if vec_override
            if vfdims == 3
                ventral_vec = U_override[1][ cartesian_pos ], V_override[1][ cartesian_pos ], W_override[1][cartesian_pos]
                 dorsal_vec = U_override[2][ cartesian_pos ], V_override[2][ cartesian_pos ], W_override[2][cartesian_pos]
            else
                ventral_vec = U_override[1][ cartesian_pos ], V_override[1][ cartesian_pos ]
                 dorsal_vec = U_override[2][ cartesian_pos ], V_override[2][ cartesian_pos ]
            end
            # dorsal_vec  = -1 .* ventral_vec; 
            wave_params.ventral_dir  = ventral_vec
            wave_params.dorsal_dir   = dorsal_vec
            wave_params.ventral_sums = dir2sums( ventral_vec )
            wave_params.dorsal_sums  = dir2sums( dorsal_vec  )
        end

        pos = Tuple( cartesian_pos )

        # 1-. Extracting vector time-series at "pos"
        vecs_norm = extract_vectors_at( pos, norm_VFs, c_idx=c_idx )
        
        # 2-. Computing rho(t) and extracting local maxima and minima
        refd       = ref_wdot( wave_params.ventral_sums, vecs_norm..., r=1, r_left=1, r_right=1 )
        min_coords = local_minima( refd, rad=mn_rad )
        max_coords = local_maxima( refd, rad=mx_rad )
        max_coords = sort( [ min_coords..., max_coords... ] )
        min_coords = []
        for i in 1:length( max_coords ) - 1
            if max_coords[i+1] > max_coords[i] + 1
                push!( min_coords, div( max_coords[i] + max_coords[i+1], 2 )  )
            end
        end 

        # 4-. segmenting waves 
        mags  = extract_mags_at( pos, mags_VFs )
        waves = segment_waves( min_coords, max_coords, refd, mags, wave_params, vecs_norm..., 
                               v_min_len=v_min_len, d_min_len=d_min_len,
                               r_max=r_max 
                            )

        for t in 1:Ntps
            out[ t ][ pos... ] = waves[t]
        end
    end

    return out
end

begin # EXTRACTING VECTORFIELD SIZE AND "VECTOR TIME SERIES"
    
    begin # VF_size, for different vectorfield formats ###########################


        """
            Finds the size of a 2D/3D vector field given as individual components

            vfsize = VF_size( U, V )
            vfsize = VF_size( U, V, W )
        """
        function VF_size( VF::Vararg{Array{T,N}}; c_idx = 1 ) where {T,N}
            return size( VF[1] )
        end
        
        """
            Finds the size of a 2D/3D vector field given a tuple of components

            vfsize = VF_size( ( U, V ) )
            vfsize = VF_size( ( U, V, W ) )
        """
        function VF_size( VF::NTuple{ND,Array{T,N}}; c_idx = 1 ) where {T,N,ND}
            return size( VF[1] )
        end
        
        """
            Finds the size of a 2D/3D vector field given a vector of components

            vfsize = VF_size( [ U, V ] )
            vfsize = VF_size( [ U, V, W ] )
        """
        function VF_size( VF::Vector{Array{T,N}}; c_idx = 1 ) where {T,N}
            return size( VF[1] )
        end

        """
            Finds the size of a 2D/3D vector field given as a 3D/4D array combining all components.

            vfsize = VF_size( [ U ;;; V ] )
            vfsize = VF_size( [ U ;;;; V ;;;; W ] )
        """
        function VF_size( VF::Array{T,N}; c_idx = 1 ) where {T,N}

            # N-1, since we want to remove the dimension that holds VF components
            vfsize = ones( Int, N-1 )

            # for each dimension
            cont = 1; 
            for i in 1:N        
                # skip the dimension that holds VF components
                if ( i == c_idx )
                    continue
                end
                # add the remaining dimensions into vfsize
                vfsize[ cont ] = size( VF, i )
                cont += 1
            end

            return Tuple( vfsize )
        end
        
    end 
    
    begin # extract_vectors_at, for different vectorfield formats ################
    
        """
            Extracts vector at "pos" from a 2D+t/3D+t vector field given as individual components

            vec = extract_vector_at( ( 2, 4 ), U, V )
            vec = extract_vector_at( [ 4, 5, 6 ], U, V, W )
        """
        function extract_vector_at( pos, VF::Vararg{Array{T,N}}; 
                                    c_idx=1 ) where {T<:AbstractFloat,N}
            return [ vf[pos...] for vf in VF ]; 
        end
        
        """
            Extracts vector at "pos" from a 2D/3D vector field given a tuple of components

            vec = extract_vector_at( ( 2, 4 ), ( U, V ) )
            vec = extract_vector_at( [ 4, 5, 6 ], ( U, V, W ) )
        """
        function extract_vector_at( pos, VF::NTuple{ND,Array{T,N}}; 
                                    c_idx=1 ) where {T<:AbstractFloat,N,ND}
            return [ vf[pos...] for vf in VF ]
        end
        
        """
            Extracts vector at "pos" from a 2D/3D vector field given a vector of components

            vec = extract_vector_at( ( 2, 4 ), [ U, V ] )
            vec = extract_vector_at( [ 4, 5, 6 ], [ U, V, W ] )
        """
        function extract_vector_at( pos, VF::Vector{Array{T,N}}; 
                                    c_idx=1 ) where {T<:AbstractFloat,N}
            return [ vf[pos...] for vf in VF ] 
        end

        """
            Extracts vector at "pos" from a 2D/3D vector field given as a 3D/4D array combining all components.

            vec = extract_vector_at( [ 8, 1 ], [ U ;;; V ] )
            vec = extract_vector_at( [ 4, 5, 6 ], [ U ;;;; V ;;;; W ] )
        """
        function extract_vector_at( pos, VF::Array{T,N}; 
                                    c_idx=1 ) where {T<:AbstractFloat,N}

            # This code is better explained with an example: 
            # Imagine pos = ( 4, 5 ): want to select the vector at position (4,5) from a 3D array combining U and V, [ U ;;; V ].
            # We need to create two 3D coordinates, one for each component: pos_u = ( 1, 4, 5 ) & pos_v = ( 2, 4, 5 ), assuming 
            # that the vector fields components are stored along the first dimension, c_idx = 1.
            
            # We start by creating a 3D vector of 0s: [ 0, 0, 0 ]
            pos_N = zeros(Int,N)

            # And copy "pos" in the dimensions that aren't the "components dimension": pos_N = [ 0, 4, 5 ]. 
            pos_N[ collect(1:N) .!= c_idx ] .= pos

            # We can obtain the 3D coordinates for each component by creating a mask for the c_idx coordinate, c_mask_N = [ 1, 0, 0 ],
            # and adding it to "pos_N": pos_N .+ c_mask_N = [ 1, 4, 5 ], pos_N .* 2 .* c_mask_N = [ 2, 4, 5 ], ... 
            c_mask_N = zeros(Int,N)
            c_mask_N[ c_idx ] = 1

            # Number of components in the VF
            NC = size(VF, c_idx); 
            return [ VF[ ( pos_N .+ c .* c_mask_N)...] for c in 1:NC ]; 
        end

        """
            Extracts all vectors at position "pos" from multiple vector fields.

            U, V = extract_vectors_at( [ 4, 5 ], [ ( U, V ), ... ] )
            U, V = extract_vectors_at( [ 4, 5 ], [ [ U ;;; V ], ... ] )

            U, V, W = extract_vectors_at( [ 4, 5, 6 ], [ ( U, V, W ), ... ] )
            U, V, W = extract_vectors_at( [ 4, 5, 6 ], [ [ U ;;;; V ;;;; W ], ... ] )
        """
        function extract_vectors_at( pos, VFs::Vector{<:Any}; c_idx=1 )

            vfsize  = VF_size( VFs[1] ); 
            Ndims   = length( vfsize ); 
            vectors = extract_vector_at.( [ pos, ], VFs, c_idx=c_idx )

            return [ [ v[i] for v in vectors ] for i in 1:Ndims ]
        end
        
    end
    
    function extract_mags_at( pos, mags::Vector{<:Any}; c_idx=1 )

        return [ m[pos...] for m in mags ]
    end
    
end

begin # EXTRACTING VECTOR MAGNITUDES AND NORMALIZING VECTORS
    
    begin # VF_magnitudes & normalize_VF #########################################

        """
            mags = VF_magnitudes( U, V )
            mags = VF_magnitudes( U, V, W )

            mags_list = VF_magnitudes.( U_list, V_list )
            mags_list = VF_magnitudes.( U_list, V_list, W_list )
        """
        function VF_magnitudes( VF::Vararg{Array{T,N}}; 
                                c_idx=1 ) where {T<:AbstractFloat,N}

            # num components
            NC = length( VF );

            # initialize magnitude array with same size as the vector field
            mags = zeros( T, VF_size( VF... )... ); 

            # adding each component .^ 2 to "mags"
            for c in 1:NC
                mags .+= VF[c] .^ 2
            end

            # sqrt.( mags )
            mags .= sqrt.( mags ); 

            return mags; 
        end
        
        """
            mags = VF_magnitudes( ( U, V ) )
            mags = VF_magnitudes( ( U, V, W ) )

            mags_list = VF_magnitudes.( [ ( U, V ), ... ] )
            mags_list = VF_magnitudes.( [ ( U, V, W ), ... ] )
        """
        function VF_magnitudes( VF::NTuple{ND,Array{T,N}}; 
                                c_idx=1 ) where {T<:AbstractFloat,N,ND}
            
            return VF_magnitudes( VF..., c_idx=c_idx )
        end
        
        
        """
            mags = VF_magnitudes( [ U, V ] )
            mags = VF_magnitudes( [ U, V, W ] )

            mags_list = VF_magnitudes.( [ [ U, V ], ... ] )
            mags_list = VF_magnitudes.( [ [ U, V, W ], ... ] )
        """
        function VF_magnitudes( VF::Vector{Array{T,N}}; 
                                c_idx=1 ) where {T<:AbstractFloat,N}
            
            return VF_magnitudes( VF..., c_idx=c_idx )
        end


        """
            mags = VF_magnitudes( [ U ;;; V ;;; W ] )
            mags = VF_magnitudes( [ U ;;;; V ;;;; W ] )

            mags_list = VF_magnitudes( [ [ U ;;; V ;;; W ], ... ] )
            mags_list = VF_magnitudes( [ [ U ;;;; V ;;;; W ], ... ] )
        """
        function VF_magnitudes( VF::Array{T,N}; 
                                c_idx=1 ) where {T<:AbstractFloat,N}


            # initialize magnitude array with same size as the vector field
            mags = zeros( T, VF_size( VF )... ); 


            # RoI is something like [ 1:h, 1:w, 1:d, 1 ], [ 1, 1:h, 1:w ], ...
            RoI = [ ( i == c_idx ) ? 1 : UnitRange(1,size(VF,i)) for i in 1:N ]; 

            # adding each compoment .^ 2 to "mags"
            for c in 1:size(VF,c_idx)
                RoI[ c_idx ] = c; 
                mags .+= VF[ RoI... ] .^ 2
            end

            # sqrt.( mags )
            mags .= sqrt.( mags ); 

            return mags; 
        end

        """
            Un, Vn, Wn = normalize_VF( U, V, W ); 

            Un_list, Vn_list, Wn_list = normalize_VF.( U_list, V_list, W_list )
        """
        function normalize_VF( VF::Vararg{Array{T,N}}; 
                               c_idx=1 ) where {T<:AbstractFloat,N}

            mags = VF_magnitudes( VF... ); 

            # avoid NaN's from dividing by 0
            mags[ mags .== 0 ] .= 1; 

            return [ vf ./ mags for vf in VF ]; 
        end
        
        """
            Un, Vn, Wn = normalize_VF( ( U, V, W ) ); 

            Un_list, Vn_list, Wn_list = normalize_VF.( [ ( U, V, W ), ... ] )
        """
        function normalize_VF( VF::NTuple{ND,Array{T,N}}; 
                               c_idx=1 ) where {T<:AbstractFloat,N,ND}

            mags = VF_magnitudes( VF... ); 

            # avoid NaN's from dividing by 0
            mags[ mags .== 0 ] .= 1; 

            return [ vf ./ mags for vf in VF ]; 
        end
        
        """
            Un, Vn, Wn = normalize_VF( [ U, V, W ] ); 

            Un_list, Vn_list, Wn_list = normalize_VF.( [ [ U, V, W ], ... ] )
        """
        function normalize_VF( VF::Vector{Array{T,N}}; 
                               c_idx=1 ) where {T<:AbstractFloat,N}

            mags = VF_magnitudes( VF... ); 

            # avoid NaN's from dividing by 0
            mags[ mags .== 0 ] .= 1; 

            return [ vf ./ mags for vf in VF ]; 
        end

        """
            VFn = normalize_VF( VF, c_idx=4 ); 

            VFn_list = normalize_VF.( VF_list, c_idx=4 )
        """
        function normalize_VF( VF::Array{T,N}; 
                               c_idx=1 ) where {T<:AbstractFloat,N}


            mags = VF_magnitudes( VF, c_idx=c_idx ) 

            # avoid NaN's from dividing by 0
            mags[ mags .== 0 ] .= 1; 

            # Creating a copy of the input to be normalized.
            VFn = deepcopy( VF );

            # RoI is something like [ 1:h, 1:w, 1:d, 1 ], [ 1, 1:h, 1:w ], ...
            RoI = [ ( i == c_idx ) ? 1 : UnitRange(1,size(VF,i)) for i in 1:N ]; 

            # dividing each compoment by "mags"
            for c in 1:size(VF,c_idx)
                RoI[ c_idx ] = c
                VFn[ RoI... ] ./= mags
            end

            return VFn; 
        end
        
    end
end

begin # AVERAGE NORMALIZED DOT PRODUCTS
    
    intA_t = Vector{<:AbstractFloat}

    function integralVector( A::Vector; typ=Float32 )
        return integralVector_unsafe!( zeros(typ,length(A)+1), A )
    end

    function integralVector_unsafe!( intA::Vector{T}, A::Vector ) where {T}
        @inbounds for idx in 1:length(A)
            intA[idx+1] = intA[idx]+T(A[idx])
        end
        return intA
    end

    """
        Returns the integral sum within [L,R] of the input integral array.
    """
    i_sums( L, R, intA::intA_t ) = intA[ R + 1 ] - intA[ L ]

    """
        Returns a tuple with the integral sum within [L,R] for each input integral array.
        This function simplifies computing the integral sums for all vector fields
        components, and it generalized to 2D and 3D vector fields:

            vf_sums = i_sums( L, R, intU, intV, intM ); 
            vf_sums = i_sums( L, R, intU, intV, intW, intM ); 
    """
    function i_sums( L, R, intAs::Vararg{intA_t} )
        return i_sums.( L, R, intAs )
    end


    """
        Returns the integral wdot score between two sets of vectors. 

        The function accepts to tuples, each of which contains the integral sums for 
        each component of the sets of vectors. For instance, in 2D isums1 and isums2
        might contain ( sumU, sumV, sumM ), where sumM is the sum of magnitudes of the
        vectors. 
    """
    function iwdot( isums1, isums2 )
        return sum( isums1[1:end-1] .* isums2[1:end-1] )/( isums1[end] * isums2[end] )
    end

    """
        Sliding windows computation of the average normalized dot product between a window of the
        "vector time series", V_t, and a reference vector. This gives a measure, rho(t), that is
        locally maximized around timepoints of coherent movement in the direction of the reference
        vector. The size of the window, and so the scale of the analysis, can be modified with the 
        parameters r, r_left and r_right. 

        Instead of working directly on the components of the "vector time series", we expect one 
        integral array for each components of the "vector time series", as well as an integral array
        for the magnitudes of the "vector time series". This allows to compute sums in constant time
        for arbitrary scales of the sliding window. 
    """
    function ref_wdot_!( out::Vector{T},            # output vector for in-place operation
                         ref_sums,                  # "sum representation" of the reference vector
                         intAs::Vararg{Vector{T}};  # integral arrays for each dimension: intU, intV, [ intW ], intM
                         r       = 1,               # RHO_region = V_t[t-r:t+r]
                         r_left  = nothing,         # overwrites the "left  r": RHO_region = V_t[t-r_left:t+r ]
                         r_right = nothing          # overwrites the "right r": RHO_region = V_t[t-r:t+r_right]
                       ) where {T}    

        @assert length( out ) == ( length( intAs[1] ) - 1 ) "incompatible size of output vector and integral vectors in ref_wdot_!"

        N = length( out ); 
        r_left  = ( r_left  == nothing ) ? r : r_left; 
        r_right = ( r_right == nothing ) ? r : r_right; 

        for t in 1+r_left:N-r_right

            L , R  = t - r_left, t + r_right; 
            sums_1 = i_sums( L, R, intAs... );
            out[t] = iwdot( sums_1, ref_sums ); 
        end

        return nothing
    end

    """
        Out-of-place implementation of "ref_wdot_!", accepting multiple integral arrays as inputs.
    """
    function ref_wdot_( ref_sums,
                        intAs::Vararg{intA_t};
                        r       = 1,
                        r_left  = nothing,
                        r_right = nothing,
                       ) where {T}

        output = zeros( eltype( intAs[1] ), length( intAs[1] ) - 1 ); 
        ref_wdot_!( output, ref_sums, intAs..., r=r, r_left=r_left, r_right=r_right )
        return output
    end 

    """
        Out-of-place implementation of "ref_wdot_!", accepting vector field components (not integral arrays).
    """
    function ref_wdot( ref_sums, 
                       VF::Vararg{Vector{T}};
                       r       = 1,
                       r_left  = nothing,
                       r_right = nothing,
                       typ     = Float32
                     ) where {T}

        intVFs = integralVector.( VF, typ=typ ); 
        M      = VF_magnitudes( VF... ); 
        intM   = integralVector( M ); 

        return ref_wdot_( ref_sums, intVFs..., intM, r=r, r_left=r_left, r_right=r_right )
    end
    
end

begin # LOCAL MINIMA & MAXIMA
    
    # local minima functions

    function local_minima!( mask::Vector{Bool}, 
                            coords::Vector{<:Integer}, 
                            input::Vector{<:Real}; 
                            rad=2 )

        #   Reseting mask (all points are initialized as minima)

        mask   .= true;
        mask[1] = false; 

        #=
            The first iteration creates a "list of minima coordinates at radius 1" by 
            considering the immediate neighbours (-1 and +1) around each point. This 
            list is stored in "coords". The first "Nminima" elements of "coords" are
            the coordinates of the minima at radius 1. 

            The list of minima at radius 1 is used to compute the list of minima at 
            radius 2. This is done in-place, so "coords" gets overwritten to contain
            the "list of minima coordinates at radius 2". Notably, the minima at radius
            2 is a subset of minima at radius 1... so we only need to check the pre-computed
            minima at radius 1 instead of having to loop though all elements in the input
            vector. 

            The list of minima at radius 2 is used to compute the list of minima at
            radius 3... and so on. 
        =#
        coords .= collect( 1:length(input) );

        N = length(input) - 1;

        for r in 1:rad

            n_minima = 0; 
            this_idx = coords[1]; 

            @inbounds for i in 1:N

                #   Check if the next value is within "r" distance of the current value

                next_idx    = coords[i+1]
                isreachable = ( ( next_idx - this_idx ) <= r )
                isminimum   = false; 

                if !isreachable

                    #   Nothing changes. The current point is stays a minimum if mask[this_idx] == 1. 

                    isminimum = mask[this_idx]
                else

                    isnextsmaller = ( input[ next_idx ] < input[ this_idx ] )   
                    isnextlarger  = !isnextsmaller;

                    #   The next value cannot be a minimum if it is larger than its previous value

                    mask[next_idx] *= isnextsmaller

                    #=
                        If (mask[this_idx] == 1 AND isnextlarger ) the current value is a minimum. We record  
                        its coordinates starting from the the begining of "coords" and keep tracks of the number
                        of minima.  
                    =#

                    isminimum = mask[this_idx] && isnextlarger; 
                    mask[this_idx] = isminimum; 

                end

                coords[1+n_minima] = coords[i]; 
                n_minima += isminimum
                this_idx  = next_idx; 
            end

            #   N_minima <= N, so the iterations should get progressively faster. 

            N = n_minima; 
            ( N < 2 ) && ( break; )
        end

        return N
    end

    function local_minima( input; rad=2 )
        mask   = zeros( Bool, size( input ) ); 
        coords = zeros(  Int, size( input ) ); 
        Nmin   = local_minima!( mask, coords, input, rad=rad ); 
        return coords[1:Nmin]
    end

    # local minima functions

    function local_maxima!( mask::Vector{Bool}, 
                            coords::Vector{<:Integer}, 
                            input::Vector{<:Real}; 
                            rad=2 )

        mask   .= true;
        mask[1] = false; 
        coords .= collect( 1:length(input) );
        N       = length(input) - 1;

        for r in 1:rad

            n_maxima = 0; 
            this_idx = coords[1]; 

            @inbounds for i in 1:N

                # Check if the next value is within "r" distance of the current value
                next_idx    = coords[i+1]
                isreachable = ( ( next_idx - this_idx ) <= r )
                ismaximum   = false; 

                if !isreachable

                    # Nothing changes. The current point is stays a minimum if mask[this_idx] == 1. 
                    ismaximum = mask[this_idx]
                else

                    isnextlarger  = ( input[ next_idx ] >  input[ this_idx ] )   
                    isnextsmaller = !isnextlarger;

                    #   The next value cannot be a minimum if it is larger than its previous value

                    mask[next_idx] *= isnextlarger

                    #=
                        If (mask[this_idx] == 1 AND isnextlarger ) the current value is a minimum. We record  
                        its coordinates starting from the the begining of "coords" and keep tracks of the number
                        of minima.  
                    =#

                    ismaximum = mask[this_idx] && isnextsmaller 
                    mask[this_idx] = ismaximum; 

                end

                coords[1+n_maxima] = coords[i]; 
                n_maxima += ismaximum
                this_idx  = next_idx; 
            end

            #   N_minima <= N, so the iterations should get progressively faster. 

            N = n_maxima; 
            ( N < 2 ) && ( break; )
        end

        return N
    end

    function local_maxima( input; rad=2 )
        mask   = zeros( Bool, size( input ) ); 
        coords = zeros(  Int, size( input ) ); 
        Nmin   = local_maxima!( mask, coords, input, rad=rad ); 
        return coords[1:Nmin]
    end
    
end

begin # WAVE SEGMENTATION LOGIC

    # wave segmentation logic
    function segment_waves!( out, 
                             minima, 
                             maxima, 
                             inter,
                             wave_params::wave_parameters, 
                             intAs::Vararg{intA_t}; 
                             r_max = 10, 
                             v_min_len = 0,
                             d_min_len = 0
                            )

        N   = length( out ); 
        Nmx = length( maxima ); 
        Nmn = length( minima );
        last_pos = 0; 
        
        for i in 1:length(minima)
            
            if minima[i] <= last_pos 
                continue
            end

            t = minima[i];

            # find the surrounding inter maxima
            tmp = searchsorted( maxima, t )
            if ( ( tmp.start > Nmx )  || ( tmp.start < 1 ) || ( tmp.stop < 1 ) || ( tmp.stop > Nmx ) ) 
                continue
            end
            mx_t_prev = maxima[ tmp.stop  ]; 
            mx_t_next = maxima[ tmp.start ];

            if mx_t_prev <= last_pos 
                continue
            end

            V_range, D_range = expand_VD( mx_t_prev, mx_t_next, inter, wave_params, intAs...,
                                          r_max=r_max, v_min_len=v_min_len, d_min_len=d_min_len, mn_t=t )

            if ( length( V_range ) == 0 ) || ( length( D_range ) == 0 )
                continue
            end

            paint_VD!( out, V_range, D_range, intAs... )

            # println( "wave timepoints:", ( t, mx_t_prev, mx_t_next,  V_range, D_range ) )

            last_pos = D_range.stop
        end

        return out
    end
    
    function segment_waves( minima, 
                            maxima, 
                            inter, 
                            M2,
                            wave_params::wave_parameters, 
                            VFs::Vararg{intA_t}; 
                            r_max=10, 
                            v_min_len=0,
                            d_min_len=0 )

        intVFs = integralVector.( VFs ); 
        M      = VF_magnitudes( VFs... ); 
        intM   = integralVector( M  ); 
        intM2  = integralVector( M2 ); 
        out    = zeros( Int, length(VFs[1]) ); 
        return segment_waves!( out, minima, maxima, inter, wave_params, intVFs..., intM, intM2,  
                               r_max=r_max, v_min_len=v_min_len, d_min_len=d_min_len )
    end

    # compute the theta between "vec_sums" and the reference vectors in "wave_params.ventral_sums". 
    function theta_with_reference_vector( vec_sums, wave_params )
        theta = iwdot( vec_sums[1:end-1], wave_params.ventral_sums );
        return theta; 
    end

    function is_similar_to_reference_vector( vec_sums, wave_params )
        return theta_with_reference_vector( vec_sums, wave_params ) > wave_params.theta_r
        # return theta_with_reference_vector( vec_sums, wave_params ) > wave_params.min_ventral_cosine
    end

    # compute the theta between "vec_sums" and the reference vectors in "wave_params.ventral_sums". 
    function theta_between_vectors( vec_sums1, vec_sums2 )
        theta = iwdot( vec_sums1[1:end-1], vec_sums2[1:end-1] );
        return theta; 
    end

    # 
    function expand_phase( t,                      # timepoint around which to expand the phase segmentation
                           phase_sums,             # "sum representation" of the vectors within the initial phase segmentation
                           wave_params,            # wave parameters
                           intAs::Vararg{intA_t};  # integral arrays
                           N = length(intAs[1])-1, # number of elements in the "vector time series".
                           t0 = 1,                 # we can exclude elements from the start of the "vector time series" by setting t0 > 1.
                           tN = N,                 # we can exclude elements from the start of the "vector time series" by setting tN < N.
                           r_max     = 10,         # max distance to expand the phase segmentation
                           min_speed =  0,         # minimum speed of vectors to be added to the phase segmentation
                           min_cos   =  0          # minimum cosine (maximum angle deviation) of vectors to be added to the phase segmentation
                        )

        # initial boundaries and sums of the phase segmentation
        t_left  = t;
        t_right = t;
        p_sums  = phase_sums; 

        # expand symmetrically (one step at a time on both direction). We don't allow discontinuities. If 
        # one direction doesn't expand at one iteration, it isn't allowed to expand in the next iteration.
        for t_off in 1:r_max

            # expanding to the left
            L = max( t0, t - t_off ) 
            expand_L = ( t_left - L ) == 1; # only allow to expand in L is adjacent to the left boundary (t_left).
            if expand_L
                tmp_sums = i_sums( L, L, intAs... )
                is_valid = theta_between_vectors( tmp_sums, p_sums ) > min_cos
                is_valid = is_valid && ( tmp_sums[end] > min_speed )
                if is_valid
                    t_left = L; 
                    p_sums = i_sums( t_left, t_right, intAs... )
                end
            end

            # expanding to the right
            R = min( tN, t + t_off )
            expand_R = ( R - t_right ) == 1; # only allow to expand in R is adjacent to the right boundary (t_right).
            if expand_R
                tmp_sums = i_sums( R, R, intAs... )
                is_valid = theta_between_vectors( tmp_sums, p_sums ) > min_cos
                is_valid = is_valid && ( tmp_sums[end] > min_speed )
                if is_valid
                    t_right = R; 
                    p_sums  = i_sums( t_left, t_right, intAs... )
                end
            end

            # stop expanding if both directions have stopped expanding
            if !expand_L && !expand_R
                break
            end
        end

        return t_left, t_right
    end

    # 
    function expand_VD( mx_t_prev, 
                        mx_t_next, 
                        inter, 
                        wave_params::wave_parameters, 
                        intAs::Vararg{intA_t};
                        r_max     = 10, 
                        v_min_len = 0,
                        d_min_len = 0, 
                        mn_t      = div( mx_t_prev + mx_t_next, 2 )
                       )

        Vmedian = i_sums( mx_t_prev, mx_t_prev, intAs... )
        Dmedian = i_sums( mx_t_next, mx_t_next, intAs... )

        # Filtering logic to the median vectors
        # A) is Vmedian pointing similar to the reference vector ? 
        # B) is Dmedian pointing sufficiently away from Vmedian ? 
        A = theta_with_reference_vector( Vmedian, wave_params ) > wave_params.theta_r; 
        B = theta_between_vectors( Vmedian, Dmedian ) < wave_params.theta_VD; 

        if !( A && B )
            return 1:0, 1:0
        end

        # Expanding V and D segmentations
        # C) This is done by expanding in each direction and as longs as the intra variance is within bounds.
        V_left, V_right = expand_phase( mx_t_prev, Vmedian, wave_params, intAs..., r_max=r_max, tN=mn_t, min_speed=wave_params.V_min_speed, min_cos=wave_params.theta_V )
        D_left, D_right = expand_phase( mx_t_next, Dmedian, wave_params, intAs..., r_max=r_max, t0=mn_t, min_speed=wave_params.D_min_speed, min_cos=wave_params.theta_D )

        # D) Check that the periods are not emtpy
        V_len = V_right - V_left + 1; 
        D_len = D_right - D_left + 1; 

        D = ( V_len > v_min_len ) && ( D_len > d_min_len )
        if !D 
            return 1:0, 1:0
        end

        # Check the average speed of V and/or the total displacement V+D
        # F) Removing segmentations if the average speed in the V phase is smaller than X... 
        V_sums = i_sums( V_left, V_right, intAs... ); 
        D_sums = i_sums( D_left, D_right, intAs... ); 
        total_displacement = V_sums[end] + D_sums[end];
        V_avg_displacement = V_sums[end]/V_sums[end-1]; 
        D_avg_displacement = D_sums[end]/D_sums[end-1];
        V_D_speed_ratio    = V_avg_displacement / D_avg_displacement

        E = ( V_avg_displacement > wave_params.V_avg_speed ) && ( total_displacement > wave_params.min_displacement ) # && ( 0.75 < V_D_speed_ratio < 4 )

        if !( E )
            return 1:0, 1:0
        end

        # THe D period actually points ventrally
        max_d = -1; 
        for d in D_left:D_right
            d_sums = i_sums( d, d, intAs... ); 
            d_dot  = iwdot( d_sums[1:end-1], wave_params.top_sums )
            max_d  = max( d_dot, max_d )
        end
 
        if max_d >= wave_params.max_top_cosine
            return 1:0, 1:0
        end

        return V_left:V_right, D_left:D_right
    end
    
    #
    function paint_VD!( out, V_range, D_range, intAs::Vararg{intA_t} )

        V_left, V_right = V_range.start, V_range.stop
        D_left, D_right = D_range.start, D_range.stop
        D_left = max( D_left, V_right+1 )
        V_sums = i_sums( V_left, V_right, intAs... ); 
        D_sums = i_sums( D_left, D_right, intAs... ); 
        out[ V_left:V_right ] .=  1
        out[ D_left:D_right ] .= -1 

        # Labelling the interface
        interface = V_right+1:D_left-1
        for ti in interface
            iface_vec = i_sums( ti, ti, intAs... );
            V_score   = iwdot( iface_vec, V_sums )
            D_score   = iwdot( iface_vec, D_sums )
            out[ ti ] = ( V_score > D_score ) ? 1 : - 1; 
        end

        return nothing
    end
    
    
end


