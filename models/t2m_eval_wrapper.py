from models.t2m_eval_modules import *
from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        map_location = torch.device('mps')
    elif torch.cuda.is_available():
        map_location = torch.device(f"cuda:{opt.gpu_id}")
    else:
        map_location = torch.device('cpu')
    
    try:    
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'text_mot_match', 'model', 'finest.tar'),
                                map_location=map_location, weights_only=True)
    except:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, 'text_mot_match', 'model', 'finest.tar'),
                                map_location=map_location)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):
        opt.dim_pose = 263
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.dim_coemb_hidden = 512
        
        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.motion_encoder.eval()
        self.movement_encoder.eval()

    
    def get_motion_embeddings_ordered(self, motions, m_lens):
        with torch.no_grad():
            # Capture the original order
            original_idx = torch.arange(len(motions))

            # Move tensors to the correct device and data type
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device)

            # Sort by sequence lengths in descending order
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            original_idx = original_idx[align_idx]  # Adjust original indices to align with sorted order

            # Perform movement encoding and motion embedding
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            # Reorder embeddings back to original order
            _, inverse_idx = torch.sort(original_idx)
            motion_embedding = motion_embedding[inverse_idx]  # Reorder to original input order

        return motion_embedding
