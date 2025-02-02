    !-----------------------------------------------------
    ! material property user defined function for ELMER:
    ! Written By: Anil Kunwar (2025-02-02)
    ! Thermal conductivity of tin as a function of temperature
    ! (kth_sn)solid = A*T + B, where A = -0.0650 and B = 106.4985
    ! Determined via differential evolution algorithm for fitting two datasets taken from the following references
    ! Manasijevic et al., 2021, Solid State Sciences, Vol. 119,106685
    ! Fatma Meydaneri and Buket Saatci IJERD (2010), Vol. Volume: 2 Issue: 1
    ! (kth_sn)liquid = 1.16*(A*T + B)
    ! Fatma Meydaneri and Buket Saatci IJERD (2010), Vol. Volume: 2 Issue: 1
    !-----------------------------------------------------
    FUNCTION getThermalConductivity( model, n, temp ) RESULT(thcondt)
    ! modules needed
    USE DefUtils
    IMPLICIT None
    ! variables in function header
    TYPE(Model_t) :: model
    INTEGER :: n
    REAL(KIND=dp) :: temp, thcondt

    ! variables needed inside function
    REAL(KIND=dp) :: refThCond, alpha,refTemp
    Logical :: GotIt
    TYPE(ValueList_t), POINTER :: material

    ! get pointer on list for material
    material => GetMaterial()
    IF (.NOT. ASSOCIATED(material)) THEN
    CALL Fatal('getThermalConductivity', 'No material found')
    END IF

    ! read in reference density at reference temperature
    refThCond = GetConstReal( material, 'Reference Thermal Conductivity Sn3.0Ag0.5Cu',GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Reference Thermal Conductivity not found')
    END IF

    ! read in Temperature Coefficient of Resistance
    alpha = GetConstReal( material, 'slope A related to Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'slope of thermal conductivity-temperature curve not found')
    END IF

    ! read in reference temperature
    refTemp = GetConstReal( material, 'Melting Point Temperature Sn3.0Ag0.5Cu', GotIt)
    IF(.NOT. GotIt) THEN
    CALL Fatal('getThermalConductivity', 'Reference Temperature not found')
    END IF


    ! compute density conductivity
    IF (refTemp <= temp) THEN ! check for physical reasonable temperature
       CALL Warn('getThermalConductivity', 'The Sn3.0Ag0.5Cu material is in liquid state.')
            !CALL Warn('getThermalConductivity', 'Using density reference value')
    !thcondt = 1.11*(refThCond + alpha*(temp))
    thcondt = 1.16*(refThCond + alpha*(temp))
    ELSE
    thcondt = refThCond + alpha*(temp)
    END IF

    END FUNCTION getThermalConductivity

