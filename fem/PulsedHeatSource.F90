FUNCTION PulsedGaussianHeatSource( Model, n, t ) RESULT(f)
  USE DefUtils

  IMPLICIT NONE

  TYPE(Model_t) :: Model
  INTEGER :: n
  REAL(KIND=dp) :: t, f

  INTEGER :: timestep, prevtimestep = -1
  REAL(KIND=dp) :: Alpha, Coeff, Speed, Dist, Dist0, &
      Time, x, y, z, s, r, yzero, phi, Frequency, PhaseShift, HarmonicSum
  REAL(KIND=dp), PARAMETER :: PI = 3.14159265358979323846_dp
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  
  SAVE Mesh, Params, prevtimestep, time, Alpha, Coeff, Speed, Dist, &
      Dist0, PulseWidth
  
  timestep = GetTimestep()
  NewTimestep = ( timestep /= prevtimestep )

  IF( NewTimestep ) THEN
    Mesh => GetMesh()
    Params => Model % Simulation
    time = GetTime()
    Alpha = GetCReal(Params,'Heat source width')
    Coeff = GetCReal(Params,'Heat source coefficient')
    Speed = GetCReal(Params,'Heat source speed')
    Dist = GetCReal(Params,'Heat source distance')
    Dist0 = GetCReal(Params,'Heat source initial position', Found)
    yzero = GetCReal(Params,'y coordinate initial position', Found)
    Frequency = GetCReal(Params,'Sine wave frequency', Found)     ! Frequency of sine wave
    PhaseShift = GetCReal(Params,'Sine wave phase shift', Found)  ! Phase shift of sine wave
    prevtimestep = timestep
  END IF

  x = Mesh % Nodes % x(n)   
  y = Mesh % Nodes % y(n)   
  z = Mesh % Nodes % z(n)   

  s = Dist0 + time * Speed  
  r = SQRT((x-s)**2 + (y-yzero)**2)

  ! Define the odd harmonics sine wave modulation function phi
  !HarmonicSum = 0.0_dp
  !DO k = 1, 5, 2  ! Sum the first 3 odd harmonics (1, 3, 5)
  !   HarmonicSum = HarmonicSum + SIN(k * Frequency * t + PhaseShift) / k
  !END DO

  ! Calculate the heat source intensity with the modulation from odd harmonics
  !phi = HarmonicSum
  ! Define the odd harmonics sine wave modulation function phi
  HarmonicSum = 0.0_dp
  IF (MOD(t, 1.0_dp / Frequency) < PulseWidth) THEN
     ! Sum the first 3 odd harmonics (1, 3, 5) while pulse is on
     DO k = 1, 5, 2  ! Sum the first 3 odd harmonics (1, 3, 5)
         HarmonicSum = HarmonicSum + SIN(k * Frequency * t + PhaseShift) / REAL(k, KIND=dp)
     END DO
     ! Set phi to HarmonicSum when pulse is on
     phi = HarmonicSum
  ELSE
    ! Set phi to 0 when pulse is off
    phi = 0.0_dp
  END IF


  f = phi * Coeff * EXP( -2*r**2 / Alpha**2 )
  
END FUNCTION GaussianHeatSource

