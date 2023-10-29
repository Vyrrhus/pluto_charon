!========================================================================
                 PROGRAM FREQ_MAP_ANALYSIS
!========================================================================
!> Analyse en fréquence fine d'une fonction temporelle complexe
!========================================================================

IMPLICIT NONE

! PRECISION & PI
INTEGER, PARAMETER       :: xp = SELECTED_REAL_KIND(15)           ! Précision des calculs
REAL(KIND=xp), PARAMETER :: PI = 3.1415926535897932384626433_xp   ! \(\pi\)

! FONCTION INPUT
REAL(KIND=xp), DIMENSION(:), ALLOCATABLE :: REEL    ! Partie réelle
REAL(KIND=xp), DIMENSION(:), ALLOCATABLE :: IMAG    ! Partie imaginaire
REAL(KIND=xp) :: t0, dT                             ! Date initiale et pas de temps
INTEGER ::  Nse                                     ! Nombre de points de la fonction
INTEGER ::  Lbd

! PARAMETRES DE RECHERCHE DES FREQUENCES PROPRES
CHARACTER(len=100) :: arg           ! arguments de commande
REAL(KIND=xp) :: ampMin, epsBruit   ! amplitude min / bruit résidual min recherchés
INTEGER :: NMax                     ! nombre de termes maximal recherché
INTEGER :: argCount                 ! nombre d'arguments passés à l'exécution
INTEGER :: jj

! FICHIERS
CHARACTER(*), PARAMETER :: INDATA="./data/FMA/input.dat"//achar(0)
CHARACTER(*), PARAMETER :: OUTDATA="./data/FMA/output.dat"//achar(0)
INTEGER :: INPUT_UNT  ! indices des fichiers
INTEGER :: IOS        ! ios

INTEGER :: ii

!------------------------------------------------------------------------------

! Ouverture fichier
OPEN(newunit=INPUT_UNT, file=INDATA, status='old', action='read', iostat=IOS)
IF (IOS /= 0)   STOP "Erreur d'ouverture du fichier"

! Allocation des tableaux
READ(INPUT_UNT, iostat=IOS, fmt=*) Nse, t0, dT  ! Paramètres de la série
ALLOCATE(REEL(Nse))
ALLOCATE(IMAG(Nse))

! Lecture
DO ii=1, Nse
    READ(INPUT_UNT, iostat=ios, fmt=*)  REEL(ii), IMAG(ii)
ENDDO
CLOSE(INPUT_UNT)

!------------------------------------------------------------------------------

! Paramètres analyse en fréquence (arguments ou valeurs par défaut)
argCount = COMMAND_ARGUMENT_COUNT()

! Valeurs par défaut
NMax     = 17       ! nombre de termes maximal recherché
ampMin   = 0._xp    ! amplitude minimale recherchée
epsBruit = 0._xp    ! bruit résiduel minimal recherché
Lbd       = 0       ! si l'on souhaite également ajuster une droite 
                    ! à la partie imaginaire (/=0 si oui)

! Valeurs en argument
IF (COMMAND_ARGUMENT_COUNT() > 0) THEN
    DO jj = 1, COMMAND_ARGUMENT_COUNT()
        CALL GET_COMMAND_ARGUMENT(jj, value=arg)
        IF (jj == 1)     READ(arg, *)   NMax
        IF (jj == 2)     READ(arg, *)   ampMin
        IF (jj == 3)     READ(arg, *)   epsBruit

    ENDDO
ENDIF

! Analyse
CALL runFreqAnalys_fort(Nse,t0,Dt,REEL,IMAG,NMax,epsBruit,ampMin,Lbd,OUTDATA)
DEALLOCATE(REEL, IMAG)

!==============================================================================
END PROGRAM FREQ_MAP_ANALYSIS
!==============================================================================