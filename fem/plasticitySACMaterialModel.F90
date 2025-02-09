    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Elastic modulus of solid alloy as piecewise function of vonMises stress
    ! yield strength (sigma_yield) is obtained from experiments
    ! When sigma <= sigma_yield , E = E_0 = 40.0 GPa   (T=298.0 K)
    ! When sigma > sigma_yield , E = sigma/((sigma/K_H))**(1/n) (T=298.0 K)
    ! Hollomon parameters
    ! where, strength coefficient K_H = 381.08 MPa and strain hardening coefficient = n = 0.196; m = 1/n = 5.1
    ! Nguyen and Park, Microelectronics Reliability 51 (2011) 1385-1392.
    ! https://www.sciencedirect.com/science/article/pii/S0026271411000953?via%3Dihub
    ! Rao et al., 2011 MSEA,
    ! https://www.sciencedirect.com/science/article/abs/pii/S0921509311001481?via%3Dihub
    ! Battau et al.
    ! https://cdn2.hubspot.net/hubfs/1871852/DfR_Solutions_Website/Resources-Archived/Publications/2002-2004/2004_FlexCrackPb-Free2_Hillman-Blattau.pdf?t=1482271815148
    ! T dependence of elasticity E(T) = As*(T-273)^2 + Bs*(T-273) + Cs , where As= -1.86E+5 Pa/K^2 , Bs = 7.5E+7 Pa/K , Cs =43.4E+09 Pa (Vianco et al., 2023)
    ! References: Vianco et al., Journal of Electronic Materials 52 (2023) 2116–2138.
    ! https://link.springer.com/article/10.1007/s11664-022-10159-y
    !-----------------------------------------------------
    FUNCTION getPlasticity( model, n, stress, temp ) RESULT(elast)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: stress, elast

    ! variables needed inside function
    REAL(KIND=dp) :: refElast, yieldsigma,   &
    strengthcoeff, mcoeff, temp, alphas, betas
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getPlasticity', 'No material found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    alphas = GetConstReal( material, 'T2 Coeff As Solid Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'T2 term of Thermal Expansivity-temperature curve solid not found')
    END IF
    
    ! read in Temperature Coefficient of Resistance
    betas = GetConstReal( material, 'T1 Coeff Bs Solid Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'Slope  term of solid Sn3.0Ag0.5Cu not found')
    END IF
    
    ! read in reference elasticity at reference temperature
    refElast = GetConstReal( material, 'Isotropic elastic modulus of Sn3.0Ag0.5Cu at 298 K in Pa',GotIt)
    !refDenst = GetConstReal( material, 'Solid_ti_rho_constant',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'Constant Youngs modulus not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    yieldsigma = GetConstReal( material, 'Yield strength of Sn3.0Ag0.5Cu materials in Pa', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'Sigma yield not found')
    END IF
     
    ! read in pseudo reference conductivity at reference temperature of liquid
    strengthcoeff = GetConstReal( material, 'Strength coefficient in Ramberg-Osgood equation',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', 'Strength Coefficient of Sn3.0Ag0.5Cu materials not found')
    END IF

    ! read in reference temperature
    mcoeff = GetConstReal( material, 'Reciprocal of strain hardening coefficient', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getPlasticity', '1/n factor not found')
    END IF
    
    ! read in the temperature scaling factor
    !tscaler = GetConstReal( material, 'Tscaler', GotIt)
    !IF(.NOT. GotIt) THEN
    !CALL Fatal('getThermalConductivity', 'Scaling Factor for T not found')
    !END IF

    ! compute density conductivity
    IF (yieldsigma <= stress) THEN ! check for physical reasonable temperature
       CALL Warn('getPlasticity', 'The Sn3.0Ag0.5Cu material undergoing plastic deformation.')
            !CALL Warn('getPlasticity', 'Using density reference value')
    !denst = 1.11*(refDenst + alpha*(temp))
    elast =  stress/(stress/strengthcoeff)**mcoeff + alphas*((temp-273.0))**2 + betas*(temp-273.0)
    ELSE
    elast =  refElast+ alphas*((temp-273.0))**2 + betas*(temp-273.0)
    END IF

    END FUNCTION getPlasticity

