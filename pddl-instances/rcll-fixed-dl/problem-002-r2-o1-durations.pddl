(define (problem rcll-production-002-durative)
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
		(order-base-color o1 BASE_RED)
		(order-cap-color o1 CAP_BLACK)
		(order-gate o1 GATE-1)
		(= (path-length C-BS INPUT C-BS OUTPUT) 3.073613)
		(= (path-length C-BS INPUT C-CS1 INPUT) 6.058649)
		(= (path-length C-BS INPUT C-CS1 OUTPUT) 9.322469)
		(= (path-length C-BS INPUT C-CS2 INPUT) 6.496925)
		(= (path-length C-BS INPUT C-CS2 OUTPUT) 7.541513)
		(= (path-length C-BS INPUT C-DS INPUT) 8.269358)
		(= (path-length C-BS INPUT C-DS OUTPUT) 6.129328)
		(= (path-length C-BS OUTPUT C-BS INPUT) 3.073613)
		(= (path-length C-BS OUTPUT C-CS1 INPUT) 7.452332)
		(= (path-length C-BS OUTPUT C-CS1 OUTPUT) 9.324498)
		(= (path-length C-BS OUTPUT C-CS2 INPUT) 4.660185)
		(= (path-length C-BS OUTPUT C-CS2 OUTPUT) 5.704772)
		(= (path-length C-BS OUTPUT C-DS INPUT) 7.731637)
		(= (path-length C-BS OUTPUT C-DS OUTPUT) 5.861686)
		(= (path-length C-CS1 INPUT C-BS INPUT) 6.058649)
		(= (path-length C-CS1 INPUT C-BS OUTPUT) 7.452333)
		(= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.863633)
		(= (path-length C-CS1 INPUT C-CS2 INPUT) 5.926527)
		(= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.475821)
		(= (path-length C-CS1 INPUT C-DS INPUT) 4.778152)
		(= (path-length C-CS1 INPUT C-DS OUTPUT) 3.914149)
		(= (path-length C-CS1 OUTPUT C-BS INPUT) 9.322468)
		(= (path-length C-CS1 OUTPUT C-BS OUTPUT) 9.324499)
		(= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.863632)
		(= (path-length C-CS1 OUTPUT C-CS2 INPUT) 6.63941)
		(= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.188704)
		(= (path-length C-CS1 OUTPUT C-DS INPUT) 5.491035)
		(= (path-length C-CS1 OUTPUT C-DS OUTPUT) 4.627031)
		(= (path-length C-CS2 INPUT C-BS INPUT) 6.496925)
		(= (path-length C-CS2 INPUT C-BS OUTPUT) 4.660184)
		(= (path-length C-CS2 INPUT C-CS1 INPUT) 5.926527)
		(= (path-length C-CS2 INPUT C-CS1 OUTPUT) 6.63941)
		(= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.352873)
		(= (path-length C-CS2 INPUT C-DS INPUT) 3.711692)
		(= (path-length C-CS2 INPUT C-DS OUTPUT) 3.176598)
		(= (path-length C-CS2 OUTPUT C-BS INPUT) 7.541513)
		(= (path-length C-CS2 OUTPUT C-BS OUTPUT) 5.704772)
		(= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.475821)
		(= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.188703)
		(= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.352873)
		(= (path-length C-CS2 OUTPUT C-DS INPUT) 2.255063)
		(= (path-length C-CS2 OUTPUT C-DS OUTPUT) 3.725891)
		(= (path-length C-DS INPUT C-BS INPUT) 8.269358)
		(= (path-length C-DS INPUT C-BS OUTPUT) 7.731637)
		(= (path-length C-DS INPUT C-CS1 INPUT) 4.778153)
		(= (path-length C-DS INPUT C-CS1 OUTPUT) 5.491036)
		(= (path-length C-DS INPUT C-CS2 INPUT) 3.711691)
		(= (path-length C-DS INPUT C-CS2 OUTPUT) 2.255063)
		(= (path-length C-DS INPUT C-DS OUTPUT) 4.084709)
		(= (path-length C-DS OUTPUT C-BS INPUT) 6.129328)
		(= (path-length C-DS OUTPUT C-BS OUTPUT) 5.861686)
		(= (path-length C-DS OUTPUT C-CS1 INPUT) 3.914149)
		(= (path-length C-DS OUTPUT C-CS1 OUTPUT) 4.627032)
		(= (path-length C-DS OUTPUT C-CS2 INPUT) 3.176598)
		(= (path-length C-DS OUTPUT C-CS2 OUTPUT) 3.725891)
		(= (path-length C-DS OUTPUT C-DS INPUT) 4.084709)
		(= (path-length START INPUT C-BS INPUT) 1.783161)
		(= (path-length START INPUT C-BS OUTPUT) 4.145949)
		(= (path-length START INPUT C-CS1 INPUT) 4.377451)
		(= (path-length START INPUT C-CS1 OUTPUT) 7.641272)
		(= (path-length START INPUT C-CS2 INPUT) 5.987144)
		(= (path-length START INPUT C-CS2 OUTPUT) 6.969337)
		(= (path-length START INPUT C-DS INPUT) 7.328156)
		(= (path-length START INPUT C-DS OUTPUT) 5.188126)
	)
	(:goal (order-fulfilled o1))
)