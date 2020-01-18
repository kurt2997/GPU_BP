# Makefile for ...su/main

include $(CWPROOT)/src/Makefile.config
CC=nvcc
CFLAGS = $(CWP_FLAGS) -D_BSD_SOURCE -D_POSIX_SOURCE -I/$I -I/usr/local/CWP/43R3/src/Complex/include
#CFLAGS = $(CWP_FLAGS) -D_BSD_SOURCE -D_POSIX_SOURCE -I$I 

D = $L/libcwp.a $L/libpar.a $L/libsu.a

B=./


LFLAGS= $(PRELFLAGS) -L$L -lsu -lpar -lcwp -lcublas -llapack -lm -lgfortran $(POSTLFLAGS)

PROGS =                 \
        $B/crg_REFINV_BP_GPU      \

#PROGS =			\
#	$B/test_GPU_backup221013 \ 


INSTALL	:	$(PROGS)
	@-rm -f INSTALL
	@touch $@


$(PROGS):	$D 
	-$(CC) $(CFLAGS) $(@F).cu $(LFLAGS) -o $@
	@$(MCHMODLINE)
	@echo $(@F) installed in $B

remake	:
	-rm -f $(PROGS) INSTALL
	$(MAKE) 
	
clean::
	rm -f a.out junk* JUNK* core
