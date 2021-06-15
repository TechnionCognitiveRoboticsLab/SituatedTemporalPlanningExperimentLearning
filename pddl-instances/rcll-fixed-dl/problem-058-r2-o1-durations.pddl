(define (problem rcll-production-058-durative)
	(:domain rcll-production-durative)
	(:objects R-1 - robot R-2 - robot o1 - order wp1 - workpiece cg1 - cap-carrier cg2 - cap-carrier cg3 - cap-carrier cb1 - cap-carrier cb2 - cap-carrier cb3 - cap-carrier C-BS - mps C-CS1 - mps C-CS2 - mps C-DS - mps CYAN - team-color)
	(:init 
		(order-delivery-window-open o1)
		(at 150 (not (order-delivery-window-open o1)))
		(can-commit-for-ontime-delivery o1)
		(at 15 (not (can-commit-for-ontime-delivery o1)))
		(mps-type C-BS BS)
		(mps-type C-CS1 CS)
		(mps-type C-CS2 CS)
		(mps-type C-DS DS)
		(location-free START INPUT)
		(location-free C-BS INPUT)
		(location-free C-BS OUTPUT)
		(location-free C-CS1 INPUT)
		(location-free C-CS1 OUTPUT)
		(location-free C-CS2 INPUT)
		(location-free C-CS2 OUTPUT)
		(location-free C-DS INPUT)
		(location-free C-DS OUTPUT)
		(cs-can-perform C-CS1 CS_RETRIEVE)
		(cs-can-perform C-CS2 CS_RETRIEVE)
		(cs-free C-CS1)
		(cs-free C-CS2)
		(wp-base-color wp1 BASE_NONE)
		(wp-cap-color wp1 CAP_NONE)
		(wp-ring1-color wp1 RING_NONE)
		(wp-ring2-color wp1 RING_NONE)
		(wp-ring3-color wp1 RING_NONE)
		(wp-unused wp1)
		(robot-waiting R-1)
		(robot-waiting R-2)
		(mps-state C-BS IDLE)
		(mps-state C-CS1 IDLE)
		(mps-state C-CS2 IDLE)
		(mps-state C-DS IDLE)
		(wp-cap-color cg1 CAP_GREY)
		(wp-cap-color cg2 CAP_GREY)
		(wp-cap-color cg3 CAP_GREY)
		(wp-on-shelf cg1 C-CS1 LEFT)
		(wp-on-shelf cg2 C-CS1 MIDDLE)
		(wp-on-shelf cg3 C-CS1 RIGHT)
		(wp-cap-color cb1 CAP_BLACK)
		(wp-cap-color cb2 CAP_BLACK)
		(wp-cap-color cb3 CAP_BLACK)
		(wp-on-shelf cb1 C-CS2 LEFT)
		(wp-on-shelf cb2 C-CS2 MIDDLE)
		(wp-on-shelf cb3 C-CS2 RIGHT)
		(order-complexity o1 c0)
		(order-base-color o1 BASE_SILVER)
		(order-cap-color o1 CAP_GREY)
		(order-gate o1 GATE-2)
		(= (path-length C-BS INPUT C-BS OUTPUT) 2.783584)
		(= (path-length C-BS INPUT C-CS1 INPUT) 6.439496)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 6.373029)
		(= (path-length C-BS INPUT C-CS2 INPUT) 5.154797)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 6.77062)
		(= (path-length C-BS INPUT C-DS INPUT) 6.114498)
		(= (path-length C-BS INPUT C-DS OUTPUT) 5.136752)
		(= (path-length C-BS OUTPUT C-BS INPUT) 2.783584)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 8.420912)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 7.095957)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 7.145549)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 8.752036)
		(= (path-length C-BS OUTPUT C-DS INPUT) 8.105251)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 7.127505)
		(= (path-length C-CS1 INPUT C-BS INPUT) 6.439495)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 8.420911)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.045268)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 5.653124)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 2.988066)
		(= (path-length C-CS1 INPUT C-DS INPUT) 5.067993)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 4.50234)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 6.373028)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 7.095956)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.045268)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.343543)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 4.678484)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 6.758411)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 6.192758)
		(= (path-length C-CS2 INPUT C-BS INPUT) 5.154796)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 7.145548)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 5.653124)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.343543)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.195783)
		(= (path-length C-CS2 INPUT C-DS INPUT) 4.385998)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 1.458824)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 6.770619)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 8.752034)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 2.988066)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 4.678484)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.195783)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 2.610652)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 2.044998)
		(= (path-length C-DS INPUT C-BS INPUT) 6.114498)
		(= (path-length C-DS INPUT C-BS OUTPUT) 8.105249)
		(= (path-length C-DS INPUT C-CS1 INPUT) 5.067994)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 6.758413)
		(= (path-length C-DS INPUT C-CS2 INPUT) 4.385999)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 2.610652)
		(= (path-length C-DS INPUT C-DS OUTPUT) 3.235214)
		(= (path-length C-DS OUTPUT C-BS INPUT) 5.136752)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 7.127504)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 4.502339)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 6.192759)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 1.458824)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 2.044998)
		(= (path-length C-DS OUTPUT C-DS INPUT) 3.235214)
		(= (path-length START INPUT C-BS INPUT) 1.436507)
		(= (path-length START INPUT C-BS OUTPUT) 2.159435)
		(= (path-length START INPUT C-CS1 INPUT) 6.330243)
		(= (path-length START INPUT C-CS1 OUTPUT) 5.005289)
		(= (path-length START INPUT C-CS2 INPUT) 5.798472)
		(= (path-length START INPUT C-CS2 OUTPUT) 6.661366)
		(= (path-length START INPUT C-DS INPUT) 6.758174)
		(= (path-length START INPUT C-DS OUTPUT) 5.780428)
	)
	(:goal (order-fulfilled o1))
)