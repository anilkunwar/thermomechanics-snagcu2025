FUNCTION PulsedGaussianHeatSource( Model, n, t ) RESULT(f)
  USE DefUtils

  IMPLICIT NONE

  TYPE(Model_t) :: Model
  INTEGER :: n
  REAL(KIND=dp) :: t, f

  INTEGER :: timestep, prevtimestep = -1
  INTEGER :: k
  REAL(KIND=dp) :: Alpha, Energy, Speed, xdist, xdist0, &
      Time, x, y, z, s, r, yzero, phiharmonic, Frequency, PhaseShift, &
      HarmonicSum, Pavg, Peffective, Coeff, Radiusball, PulseWidth
  !REAL(KIND=dp), PARAMETER :: PI = 3.14159265358979323846_dp
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  LOGICAL :: Found, NewTimestep
  
  SAVE Mesh, Params, prevtimestep, time, Alpha, Energy, Speed, xdist,  &
      xdist0, PulseWidth, HarmonicSum, Pavg, Peffective, Coeff, Radiusball, &
      phiharmonic, Frequency, PhaseShift
  
  timestep = GetTimestep()
  NewTimestep = ( timestep /= prevtimestep )

  IF( NewTimestep ) THEN
    Mesh => GetMesh()
    Params => Model % Simulation
    time = GetTime()
    Alpha = GetCReal(Params,'Heat source half-width')
    PulseWidth = GetCReal(Params,'Laser pulsewidth')
    PhaseShift = GetCReal(Params,'Phase shift for sine wave')
    Energy = GetCReal(Params,'Energy per pulse in J')
    Radiusball = GetCReal(Params,'Radius of solder ball')
    !Coeff = GetCReal(Params,'Heat source coefficient')
    Speed = GetCReal(Params,'Heat source speed')
    xdist = GetCReal(Params,'Heat source distance x')
    xdist0 = GetCReal(Params,'Heat source initial position x', Found)
    yzero = GetCReal(Params,'y coordinate initial position', Found)
    !zzero = GetCReal(Params,'z coordinate initial position', Found)
    Frequency = GetCReal(Params,'Repetition rate', Found)     ! Frequency of sine wave
    !PhaseShift = GetCReal(Params,'Sine wave phase shift', Found)  ! Phase shift of sine wave
    prevtimestep = timestep
  END IF

  x = Mesh % Nodes % x(n)   
  y = Mesh % Nodes % y(n)   
  z = Mesh % Nodes % z(n)   

  s = xdist0 + time * Speed  
  !r = SQRT((x-s)**2 + (y-yzero)**2)
  r = SQRT((x-s)**2 + (y-s)**2) ! since the laser doesnot move, s can be put at both place

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
     phiharmonic = HarmonicSum
  ELSE
    ! Set phi to 0 when pulse is off
    phiharmonic = 0.0_dp
  END IF
  
  ! Peak power P_peak = Energy/pulse_width = 7.5E-3/6E-9 W ! Not used in this expression
  ! Average power. Power of a continuous wave laser 
  Pavg = 1.0E-03*Energy*Frequency !W
  !arearec = 3.46E-03  ! ratio of solder surface area to beam area
  Peffective = 2*Pavg*(radiusball/Alpha)**2 !the factor 2 applies for frequency of 50 Hz and pulse width of 0.01 ms
  Coeff = 2*Peffective/(3.14*radiusball**2)


  f = phi * Coeff * EXP( -2*r**2 / Alpha**2 )
  
END FUNCTION GaussianHeatSource

