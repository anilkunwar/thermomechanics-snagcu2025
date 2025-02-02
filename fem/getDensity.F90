    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Density of solid  SnAgCu fitted as a function of temperature
    ! (rho_alloy)solid = As*(T+248) +  Cs, where As = -0.000476 kg/m3K  and Cs = 7270.0 kg/m3
    ! 298 < T < Tm where Tm = 492.0 K
    ! Reference: Wang and Xian, JOurnal of Electronic Materials, 34 (2005) 1414-1419.
    ! https://link.springer.com/article/10.1007/s11664-005-0199-x
    ! Density of liquid SnAgCu fitted as a function of temperature
    !(rho_alloy)liquid  = Al*(T+273) + Cl, where Al = -0.691 kg/m3K  and Cl = 7457.7 kg/m3 (492.0 K < T < 2500.0 K) 7.4577-0.000691T
    ! Reference: Gasior et al., Journal of Phase Equilibria and Diffusion, 25 (2004) 115-121.
    ! https://link.springer.com/article/10.1007/s11669-004-0003-2
    !-----------------------------------------------------
    FUNCTION getDensity( model, n, temp ) RESULT(denst)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, denst, tscaler

    ! variables needed inside function
    REAL(KIND=dp) :: refSolDenst, refLiqDenst, refTemp,  &
    alphas, alphal
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getDensity', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    refSolDenst = GetConstReal( material, 'Reference Density Solid Sn3.0Ag0.5Cu',GotIt)
    !refDenst = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getDensity', 'Reference Density Solid Sn3.0Ag0.5Cu not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    alphas = GetConstReal( material, 'Density Coeff Solid Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getDensity', 'slope of Density-temperature curve solid not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    !betas = GetConstReal( material, 'Density Coeff Bs Solid Sn3.0Ag0.5Cu', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getDensity', 'Coefficientt of Bs*T2 term solid Sn3.0Ag0.5Cu not found')
    !END IF
    
    ! read in  Ds in Ds*ln(T) term
    !deltas = GetConstReal( material, 'Density Coeff Ds Liquid Sn3.0Ag0.5Cu', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getDensity', 'Coefficient of logT term liquid Sn3.0Ag0.5Cu not found')
    !END IF
    
    ! read in reference density at reference temperature
    refLiqDenst = GetConstReal( material, 'Reference Density Liquid Sn3.0Ag0.5Cu',GotIt)
    !refDenst = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getDensity', 'Reference Density Solid Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in pseudo reference conductivity at reference temperature of liquid
    alphal = GetConstReal( material, 'Density Coefficient Liquid Sn3.0Ag0.5Cu',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getDensity', 'Density Coefficient Al of Liquid Sn3.0Ag0.5Cu not found')
    END IF

    ! read in reference temperature
    refTemp = GetConstReal( material, 'Melting Point Temperature Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getDensity', 'Reference Melting Temperature of Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in the temperature scaling factor
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Scaling Factor for T not found')
    END IF

    ! compute density conductivity
    IF (refTemp <= temp) THEN ! check for physical reasonable temperature
       CALL Warn('getDensity', 'The Sn3.0Ag0.5Cu material is in liquid state.')
            !CALL Warn('getDensity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    denst = refLiqDenst + alphal*((tscaler)*(temp+273))
    ELSE
    denst = refSolDenst + alphas*((tscaler)*(temp+248)) 
    END IF

    END FUNCTION getDensity

