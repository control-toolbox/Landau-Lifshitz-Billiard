TAP=tapenade3.8
OPTT='-tangent -fixinterface -inputlanguage fortran90 -outputlanguage fortran90'

# hfun -> hfun_d
TAPENADE_HOME=${TAP} ${TAP}/bin/tapenade ${OPTT} -O ./ \
-tgtfuncname _d -head "hfun(x, p)>(h)" \
-o hfun \
hfun.f90

# hfun_d -> hfun_d_d
TAPENADE_HOME=${TAP} ${TAP}/bin/tapenade ${OPTT} -O ./ \
-tgtfuncname _d -head "hfun_d(x, p)>(hd)" \
-o hfun_d \
hfun_d.f90

# hfun_d_d -> hfun_d_d_d
TAPENADE_HOME=${TAP} ${TAP}/bin/tapenade ${OPTT} -O ./ \
-tgtfuncname _d -head "hfun_d_d(x, p)>(hdd)" \
-o hfun_d_d \
hfun_d_d.f90

