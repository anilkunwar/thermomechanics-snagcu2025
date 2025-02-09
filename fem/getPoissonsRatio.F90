    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Nu of solid  SnAgCu fitted as a function of temperature
    ! (rho_alloy)solid = As*T^2 + Bs*T+ Cs, where As = -2.855e-07 K^-2, Bs =  1.012e-03 K^-1 and Cs =  3.995e-02 kg/m3
    ! 298 < T < Tm where Tm = 492.0 K
    ! Reference: Hassan et al., 2019, ITHERM Conference.
    ! https://ieeexplore.ieee.org/document/8757309
    !-----------------------------------------------------
    FUNCTION getNu( model, n, temp ) RESULT(pratio)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, pratio, tscaler

    ! variables needed inside function
    REAL(KIND=dp) :: refSolPratio, alphas, &
    betas, refTemp
    
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getNu', 'No material found')
    END IF

    ! read in reference conductivity at reference temperature
    refSolPratio = GetConstReal( material, 'Cs for nu Solid Sn3.0Ag0.5Cu',GotIt)
    !refPratio = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getNu', 'Cs for nu Solid Sn3.0Ag0.5Cu not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    alphas = GetConstReal( material, 'T2 Coeff Solid Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getNu', 'T2 Coeff Solid Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    betas = GetConstReal( material, 'T1 Coeff Solid Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getNu', 'Coefficientt of Bs*T2 term solid Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in  Ds in Ds*ln(T) term
    !deltas = GetConstReal( material, 'Density Coeff Ds Liquid Sn3.0Ag0.5Cu', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getNu', 'Coefficient of logT term liquid Sn3.0Ag0.5Cu not found')
    !END IF
    
    ! read in reference density at reference temperature
    !refLiqPratio = GetConstReal( material, 'Reference Density Liquid Sn3.0Ag0.5Cu',GotIt)
    !refPratio = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getNu', 'Reference Density Solid Sn3.0Ag0.5Cu not found')
    !END IF
    
    ! read in pseudo reference conductivity at reference temperature of liquid
    !alphal = GetConstReal( material, 'Density Coefficient Liquid Sn3.0Ag0.5Cu',GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getNu', 'Density Coefficient Al of Liquid Sn3.0Ag0.5Cu not found')
    !END IF

    ! read in reference temperature
    refTemp = GetConstReal( material, 'Melting Point Temperature Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getNu', 'Reference Melting Temperature of Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in the temperature scaling factor
    tscaler = GetConstReal( material, 'Tscaler', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Scaling Factor for T not found')
    END IF

    ! compute density conductivity
    IF (refTemp <= temp) THEN ! check for physical reasonable temperature
       CALL Warn('getNu', 'The Sn3.0Ag0.5Cu material is in liquid state.')
            !CALL Warn('getNu', 'Using density reference value')
    !pratio = 1.11*(refPratio + alpha*(temp))
    pratio = 1.0 ! no solid mechanics calculation in liquid
    ELSE
    pratio = refSolPratio + alphas*((tscaler)*(temp))**2 + betas*((tscaler)*(temp))
    END IF

    END FUNCTION getNu

