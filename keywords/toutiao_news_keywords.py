car_keywords = ['汽车','发动机', '摩托车', '丰田', '新车', '价格', '二手车','4s店', '奥迪', '车型', '吉利', '雷克萨斯', '电动车','新能源', '比亚迪', 
                '保时捷', '豪车', '油耗', '手动挡', '车辆', '车主', '款车', '性能', '法拉利', '宝骏', '车子', '大众', '变速箱', 
                '电动', '奇瑞', '高尔夫', '房车', '汉兰达', '哈弗', '红旗', '众泰', '底盘', '手刹', '起亚', '雪铁龙', '买车', '国产车', '奔驰','宝马']
entertainment_keywords = ['范冰冰', '电视剧', '赵丽颖', '娱乐圈', '明星', '演员', '女神', '粉丝', '刘德华', '谢娜', '迪丽', '复仇者', '戛纳', '鹿晗', '电影节', '发布会',
                          '颜值', '演唱会', '热巴', '母亲节', '郑爽', '主持人', '照片', '节目', '黄圣依', '刘亦菲', '韩国', '佟丽娅', '刘雨欣', '造型', '新剧', 
                          '张檬', '红毯', '张杰', '女明星', '李小璐', '火', '角色', '贾乃亮', '谢霆锋', '演技', '李冰冰', '电影', '女星', '男星', '女明星', '追星', '古装', 
                          '鲜肉', '张艺兴', '圈粉']
tech_keywords = ['5g', 'ai', 'app', 'iphone', 'oppo', '互联网', '京东', '人工智能', '区块', '升级', '华为', '小米', '张一鸣', '微信', '微软', '支付宝', '数据',
                 '智能', '机器人', '淘宝', '滴滴', '电商', '程序', '联想', '芯片', '苹果', '谷歌', '软件', '链', '阿里', '雷军', '马化腾', '高通']
game_keywords = ['cf', 'dnf', 'kz', 'lol', 'rng', 'uzi', '战队', '手游', '打野', '梦幻', '求生', '游戏', '版本', '王者', '玩', '玩家', '皮肤', 
                 '绝地', '联盟', '英雄', '荒野', '荣耀', '装备', '赛', '赛季', '队友', '魔兽']
sports_keywords = ['火箭', '詹姆斯', '勇士', '中国队', '猛龙', '球员', '球队', '骑士', '米切尔', '张继科', '上港', '骑士队', '泰伦卢', '马拉松', '中国女排', '米兰', '刘诗雯',
                   '世界杯', '世锦赛', '晋级', '巴萨', '主教练', '火箭队', '权健', '乔丹', '恒大', '球', '运动员', '鲁能', '赢球', '汤普森', '主场', '积分榜', '世界冠军', 
                   '进球', '霍福德', '爵士', '马德里', '西甲', '球衣', '主力阵容', '乒赛', '球星', '勇士队', '足坛', '球迷', '足球', '男篮', '奥尼尔', '易建联']
world = ['下台', '中东', '伊拉克', '伊朗核', '伊核', '制裁', '各国', '地震', '奥巴马', '宣布', '就任', '巴基斯坦', '强国', '总理', '总统',
          '普京', '欧洲', '法国', '特朗普', '联合国', '英国', '退群', '选举', '韩国', '领导人']
edu = ['上课时', '专业', '中学', '中小学', '中考', '公务员', '初中', '北大', '大学', '大学生', '学习', '学校', '学生', '学院', '家长', '小学', '幼儿园',
       '录取', '成绩', '招生', '教师', '教育', '数学', '校园', '毕业', '留学', '研究生', '老师', '考', '考生', '考研', '考试', '英语', '语文', '高中', 
       '高校', '高考']
travel = ['三亚', '云南', '公园', '古镇', '岛', '攻略', '旅游', '旅游景点', '旅行', '景区', '景点', '游客', '游玩', '美景', '美食', '自驾游', '西藏', 
          '贵州', '酒店', '门票', '风景']

agriculture = ['乡村', '养', '养殖', '养猪', '农业', '农民', '土地', '小麦', '扶贫', '振兴', '村', '村民', '果树', '猪', '猪价', '猪肉', '玉米', '生猪', 
          '种植', '脱贫', '镇']
finance = ['上市', '业绩', '亏损', '交易', '亿', '亿元', '企业', '信用卡', '净赚', '创业', '原油', '品牌', '基金', '巴菲特', '币', '市场', '散户', 
           '比特', '涨停', '炒股', '理财', '美元', '股', '股东', '股份', '股市', '股票', '营收', '融资', '货币', '资金', '赚钱', '走势', '金融', '银行', '黄金']
story = ['丈夫', '儿媳', '儿子', '前夫', '前妻', '女人', '女儿', '女子', '妈', '妈妈', '妻子','婆婆', '婆家', '婚礼', '嫁给', '嫂子', '孙子', '孩子', 
         '家', '家里', '岳母', '弟弟', '彩礼', '怀孕', '打工', '母亲', '民间故事', '父亲', '父母', '男友', '男子', '离婚', '结婚', '老人', '老公', '老婆']
military = ['二战','军事', '军事基地', '军人', '军队', '击落', '原子弹', '发射', '叙利亚', '坦克', '士兵', '子弹', '导弹', 
            '战争', '战斗', '战斗机', '战机', '打仗', '无人机', '日军', '架', '歼', '海军', '潜艇', '直升机', '美军', '航母', '苏联', '解放军', 
            '轰', '轰炸', '轰炸机', '部队', '阅兵', '飞机', '飞行员']
house = ['业主', '买房', '买房子', '二手房', '产权', '住宅', '住房', '公寓', '公积金', '套', '套房', '小区', '开发商', '房', '房产', '房价', '房地产', 
         '房子', '房贷', '新房', '楼市', '楼盘', '物业', '租房', '购房', '限购', '首付']
culture = ['一首', '上联', '下联', '书法', '传统', '作品', '博物馆', '原创', '大师', '对联', '小说', '文化', '春风', '月色', '欣赏', '求下联', 
            '红楼梦', '经典', '老子', '艺术', '芳草', '诗', '诗词', '醉吟']