import torch


def bundle_states(
        ctx,
        pre_idx, post_idx,
        state_in,
        model, integrator,
        mass_matrix_freq,
        dt, num_samples, sigma,
):
    """
    Simulate a bundle of states in a contact phase, adding noise to the specified tensors and averaging them.
    :param ctx: Context object containing the simulation state.
    :param pre_idx: Index of the pre-contact state.
    :param post_idx: Index of the post-contact state.
    :param state_in: The initial state to start the simulation from.
    :param model: The simulation model.
    :param integrator: The integrator used for simulation.
    :param mass_matrix_freq: Frequency of mass matrix updates.
    :param dt: Time step for simulation.
    :param num_samples: Number of samples to generate.
    :param sigma: Standard deviation of the noise.
    """
    bundle_len = post_idx - pre_idx + 1
    print("BUNDLINGGG")

    # Prepare bundle states
    # ctx.bundle_states = []
    ctx.bundle_states = [
        [ctx.models[s + 1].state() for _ in range(bundle_len)]
        for s in range(num_samples)
    ]
    print(ctx.bundle_states)

    # Prepare bundle controls
    bundle_control = [
        state_in.joint_act for _ in range(num_samples)
    ]
    ctx.bundle_control = torch.stack(bundle_control, dim=0).to(dtype=torch.float32, device=model.adapter)

    # Add gaussian noise to the controls (without grad?)
    # with torch.no_grad():
    noise = torch.randn_like(ctx.bundle_control) * sigma
    ctx.bundle_control = ctx.bundle_control + noise

    # Accumulators for joint_q and joint_qd
    final_state = ctx.models[0].state()
    final_joint_q = torch.zeros_like(final_state.joint_q)
    final_joint_qd = torch.zeros_like(final_state.joint_qd)

    # Forward simulation for each bundle
    for sample in range(num_samples):
        ctrl = ctx.bundle_control[sample]

        for step in range(bundle_len-1):
            print(f"Simulating sample {sample}, step {step} with control {ctrl}")
            curr = ctx.bundle_states[sample][step]
            nxt = ctx.bundle_states[sample][step + 1]

            curr.joint_act = ctrl
            integrator._simulate(
                ctx.tape,
                ctx.models[sample+1],
                curr,
                nxt,
                dt,
                update_mass_matrix=((step % mass_matrix_freq) == 0)
            )

        final_joint_q += ctx.bundle_states[sample][-1].joint_q
        final_joint_qd += ctx.bundle_states[sample][-1].joint_qd

    # Average the final bundle states into the final state
    final_joint_q /= num_samples
    final_joint_qd /= num_samples
    final_state.joint_q = final_joint_q
    final_state.joint_qd = final_joint_qd
    return final_state
