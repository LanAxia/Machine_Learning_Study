def convolve(x, filter, step=1):
    mat_height, mat_width = x.shape
    filter_height, filter_width = filter.shape
    h_range = np.arange(0, mat_height - filter_height + 1, step)
    w_range = np.arange(0, mat_width - filter_width + 1, step)
    y = np.zeros((len(h_range), len(w_range)))
    for h in h_range:
        for w in w_range:
            sub_x = x[h:h+filter_height, w:w+filter_width]
            y[h][w] = np.sum(np.multiply(sub_x, filter))
    return y


a = np.ones((2, 2, 5, 5))
filters = np.array([
    [
        [0, 1, 0], 
        [1, 0, 1], 
        [0, 1, 0], 
    ], 
    [
        [0, 1, 0], 
        [1, 0, 1], 
        [0, 1, 0], 
    ], 
])
def foreward(xs_mats):  # 4 dimension
    xs_mats = np.pad(xs_mats, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ys_mats = []
    for x_mats in xs_mats:
        y_mats = []
        for x_mat in x_mats:
            for filter in filters:
                y_mats.append(convolve(x_mat, filter, 1))
        y_mats = np.array(y_mats)
        ys_mats.append(y_mats)
    ys_mats = np.array(ys_mats)
    return ys_mats

mid_val = foreward(a)
mid_val.shape

''''''

def backward(xs_mats, trainable=True, learning_rate=0.01):
    xs_mats = np.pad(xs_mats, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ys_mats = np.zeros(a.shape)
    for x_i, x_mats in enumerate(xs_mats):
        y_height, y_width = ys_mats.shape[2:]  # y矩阵的大小
        for mat_i, x_mat in enumerate(x_mats):
            y_i =  mat_i // len(filters)  # 第i个y的矩阵
            flipped_filter = np.flip(filters[mat_i % len(filters)])
            filter_height, filter_width = flipped_filter.shape
            for y_h in range(y_height):
                for y_w in range(y_width):
                    sub_x = x_mat[y_h * 1:y_h * 1 + filter_height, y_w * 1:y_w * 1 + filter_width]
                    ys_mats[x_i][y_i][y_h][y_w] += np.tensordot(flipped_filter, sub_x)
    return ys_mats
backward(mid_val)

dcs = mid_val
filters = filters
def update(filters, learning_rate=0.01):
    filter_grad = np.zeros(filters.shape)
    os = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)))
    filter_height, filter_width = filters.shape[1:]
    o_height, o_width = os.shape[2:]
    for image_i, image_dcs in enumerate(dcs):
        for mat_i, mat in enumerate(image_dcs):
            o_i = mat_i // len(filters)
            filter_i = mat_i % len(filters)
            for h in range(filter_height):
                for w in range(filter_width):
                    h_range = np.arange(h, o_height - filter_height + h + 1, 1)
                    w_range = np.arange(w, o_width - filter_width + w + 1, 1)
                    sub_o = os[image_i][o_i][h_range][:, w_range]
                    test = np.tensordot(sub_o, mat)
                    filter_grad[filter_i][h, w] += np.tensordot(sub_o, mat)
    
    filters = filters - learning_rate * filter_grad
    return filters
update(filters, 0.01)

class Conv(Layer):
    def __init__(self, filter_num, filter_shape, step=1, padding=None):
        self.filter = np.random.randn(filter_num, filter_shape[0], filter_shape[1])
        self.filter_num = filter_num
        self.step = step
        self.foreward_mat = None
        self.backward_mat = None
        self.padding = padding
        if self.padding is None:
            self.padding = (self.filter.shape[1] - 1) // 2
    
    def activate_layer(self, input_shape):  # 卷积中输入是4 dimension，忽略bs后是3 dimension
        output_height = (input_shape[1] + 2*self.padding - self.filter.shape[1] + 1) // \
            self.step + min(1, (input_shape[1] + 2*self.padding - self.filter.shape[1] + 1) % \
            self.step)
        output_width = (input_shape[2] + 2*self.padding - self.filter.shape[2] + 1) // \
            self.step + min(1, (input_shape[2] + 2*self.padding - self.filter.shape[2] + 1) % \
            self.step)
        return (input_shape[0] * self.filter_num, output_height, output_width)

    def pad(self, x):
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))  # 使用numpy.pad函数进行padding
    
    def convolve(self, x, filter, step):
        mat_height, mat_width = x.shape
        filter_height, filter_width = filter.shape
        h_range = np.arange(0, mat_height - filter_height + 1, step)
        w_range = np.arange(0, mat_width - filter_width + 1, step)
        y = np.zeros((len(h_range), len(w_range)))
        for h in h_range:
            for w in w_range:
                sub_x = x[h:h+filter_height, w:w+filter_width]
                y[h][w] = np.sum(np.multiply(sub_x, filter))
        return y

    def foreward(self, xs_mats):  # 4 dimension
        self.os = xs_mats.copy()  # 未padding
        xs_mats = self.pad(xs_mats)
        ys_mats = []
        for x_mats in xs_mats:
            y_mats = []
            for x_mat in x_mats:
                for filter in self.filter:
                    y_mats.append(self.convolve(x_mat, filter, self.step))
            y_mats = np.array(y_mats)
            ys_mats.append(y_mats)
        ys_mats = np.array(ys_mats)
        return ys_mats
    
    def backward(self, xs_mats, trainable=True, learning_rate=0.01):
        self.dcs = xs_mats.copy()
        xs_mats = self.pad(xs_mats)
        ys_mats = np.zeros(self.os.shape)
        for x_i, x_mats in enumerate(xs_mats):
            y_height, y_width = ys_mats.shape[2:]  # y矩阵的大小
            for mat_i, x_mat in enumerate(x_mats):
                y_i =  mat_i // self.filter_num  # 第i个y的矩阵
                flipped_filter = np.flip(self.filter[mat_i % self.filter_num])  # 旋转180度
                filter_height, filter_width = flipped_filter.shape
                for y_h in range(y_height):
                    for y_w in range(y_width):
                        sub_x = x_mat[y_h * self.step:y_h * self.step + filter_height, \
                            y_w * self.step:y_w * self.step + filter_width]
                        ys_mats[x_i][y_i][y_h][y_w] += np.tensordot(flipped_filter, sub_x)
        if trainable:
            self.update(learning_rate)
        return ys_mats
    
    def update(self, learning_rate=0.01):
        filter_grad = np.zeros(self.filter.shape)
        os = self.pad(self.os)
        filter_height, filter_width = self.filter.shape[1:]
        o_height, o_width = os.shape[2:]
        for image_i, image_dcs in enumerate(self.dcs):
            for mat_i, mat in enumerate(image_dcs):
                o_i = mat_i // self.filter_num
                filter_i = mat_i % self.filter_num
                for h in range(filter_height):
                    for w in range(filter_width):
                        h_range = np.arange(h, o_height - filter_height + h +1, self.step)
                        w_range = np.arange(w, o_width - filter_width + w + 1, self.step)
                        sub_o = os[image_i][o_i][h_range][:, w_range]
                        filter_grad[filter_i][h, w] += np.tensordot(sub_o, mat)
        
        self.filter = self.filter - learning_rate * filter_grad