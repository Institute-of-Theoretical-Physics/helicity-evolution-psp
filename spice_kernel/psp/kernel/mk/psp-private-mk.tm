KPL/MK

Meta-kernel for Parker Solar Probe (PSP)
ver. 20260122
written by M. Iizawa

PSP spk file (.bsp) was downloaded from
https://cdaweb.gsfc.nasa.gov/pub/data/psp/ephemeris/spice/ephemerides/

The others were from
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/


Frame kernels : FK (tf)
Only the standard inertial frame will be used. FK is not needed.

Leapseconds Kernel : LSK (tls)
The latest LSK naif0012.tls is loaded.
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/

Spacecraft/Planet Ephemeris Kernel : SPL (bsp)
https://naif.jpl.nasa.gov/pub/naif/ULYSSES/kernels/spk/

Planetary Constants Kernel : PCK (text : tpc, binary : bpc)
Only generic text PCKs is needed.
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

Digital Shape Kernel : DSK
Detailed topographic data of celestial bodies. They are not needed.

C-Kernel : CK
Orientation of spacecraft and onboard equipment. Not disclosed.

Spacecraft Clock Kernel : SCLK
It is used for converting between the spacecraft's onboard clock and SPICE's standard time system (TDB/ET/UTC), usualy for CK. Not disclosed.

Instrument Kernel : IK
Geometric characteristics of onboard equipment. Not disclosed.

Spacecraft and Planet Kernel : SPK (bsp)
the positions and velocities of celestial bodies and spacecraft in a time series.

for PSP at naif site
https://naif.jpl.nasa.gov/pub/naif/SPP/kernels/
Note : No SPK files are available.

for Sun
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
de442s.bsp : Short ver. of DE442. DE442 is updated version of DE440 regarding some planets. Sun data are same as DE440.
https://doi.org/10.3847/1538-3881/abd414




\begindata

PATH_VALUES       = ( '..' )
PATH_SYMBOLS      = ( 'KERNELS' )
KERNELS_TO_LOAD   = (
			'$KERNELS/lsk/naif0012.tls'
			'$KERNELS/pck/pck00011.tpc'
			'$KERNELS/spk/spp_nom_20180812_20300101_v043_PostV7.bsp'
			'$KERNELS/spk/de442s.bsp'
			)




SKD_VERSION = '20260125'
MK_IDENTIFIER = 'psp-private-kernel'

