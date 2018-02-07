__author__ = 'chuntingzhou'


def evaluate(data_loader, path, model, model_name,isTest):
    # Warning: to use this function, the input should be setE.bio.conll that is consistent with the conll format
    sents, char_sents, tgt_tags, discrete_features, bc_feats = data_loader.get_data_set(path, args.lang)

    prefix = model_name + "_" + str(uid)
    # tot_acc = 0.0
    predictions = []
    gold_standards = []
    i = 0
    for sent, char_sent, tgt_tag, discrete_feature, bc_feat in zip(sents, char_sents, tgt_tags, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False)

        assert len(best_path) == len(tgt_tag)
        # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
        # tot_acc += acc
        predictions.append(best_path)
        gold_standards.append(tgt_tag)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.txt" % (prefix)
    eval_output_fname = "%s_eval_score.txt" % (prefix)
    with open(pred_output_fname, "w") as fout:
        for pred, gold in zip(predictions, gold_standards):
            for p, g in zip(pred, gold):
                fout.write("XXX " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    os.system("../eval/conlleval.v2 < %s > %s" % (pred_output_fname, eval_output_fname))

    with open(eval_output_fname, "r") as fin:
        lid = 0
        for line in fin:
            if lid == 1:
                fields = line.split(";")
                acc = float(fields[0].split(":")[1].strip()[:-1])
                precision = float(fields[1].split(":")[1].strip()[:-1])
                recall = float(fields[2].split(":")[1].strip()[:-1])
                f1 = float(fields[3].split(":")[1].strip())
            lid += 1

    output = open(eval_output_fname, "r").read().strip()
    print output
    os.system("rm %s" % (eval_output_fname,))
    if not isTest:
	os.system("rm %s" % (pred_output_fname,))

    return acc, precision, recall, f1


def evaluate_lr(data_loader, model, model_name, score_file, setE, data):
    sents, char_sents, discrete_features, origin_sents, bc_feats = data
    print "Evaluation data size: ", len(sents)
    prefix = model_name + "_" + str(uid)
    predictions = []
    i = 0
    for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False)

        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.conll" % (prefix)
    with codecs.open(pred_output_fname, "w", "utf-8") as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                if p not in data_loader.id_to_tag:
                    print "ERROR: Predicted tag not found in the id_to_tag dict, the id is: ", p
                    p = 0
                fout.write(word + "\tNNP\tNP\t" + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (prefix)
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    scoring_file = "../eval/%s_score_file" % (prefix)
    run_program(pred_output_fname, pred_darpa_output_fname, setE)

    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)
    os.system("bash %s ../eval/%s %s" % (score_file, final_darpa_output_fname, scoring_file))

    prec = 0
    recall = 0
    f1 = 0
    with codecs.open(scoring_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec=float(columns[-4])
                recall =float(columns[-3])
                f1 = float(columns[-2])
                break

    return 0, prec, recall, f1

def evaluate_lr_splitHashtag(data_loader, model, model_name, score_file, setE, args, data):
    sents, char_sents, discrete_features, origin_sents, bc_feats = data
    print "Evaluation data size: ", len(sents)
    prefix = model_name + "_" + str(uid)
    predictions = []
    i = 0
    for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False)

        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.conll" % (prefix)
    with codecs.open(pred_output_fname, "w", "utf-8") as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                if p not in data_loader.id_to_tag:
                    print "ERROR: Predicted tag not found in the id_to_tag dict, the id is: ", p
                    p = 0
                fout.write(word + "\tNNP\tNP\t" + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (prefix)
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    final_output_fname = "../eval/%s_fixed_darpa_output.conll" % (prefix)
    scoring_file = "../eval/%s_score_file" % (prefix)
    run_program(pred_output_fname, pred_darpa_output_fname, setE)

    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)

    #Putting Hashtags back
    os.system(
            "python ../scripts/fix_char_offsets.py --edl_file ../eval/%s --original_LTF_dir ../helper_files/%s/original  --split_hashtag_dir ../helper_files/%s/split_all_hashtags_v2 > ../eval/%s" % (
                final_darpa_output_fname, args.lang, args.lang, final_output_fname))
    os.system("bash %s ../eval/%s %s" % (score_file, final_output_fname, scoring_file))

    prec = 0
    recall = 0
    f1 = 0
    with codecs.open(scoring_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec=float(columns[-4])
                recall =float(columns[-3])
                f1 = float(columns[-2])
                break

    return 0, prec, recall, f1


def replace_singletons(data_loader, sents, replace_rate):
    new_batch_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if word in data_loader.singleton_words:
                new_sent.append(word if np.random.uniform(0., 1.) > replace_rate else data_loader.word_to_id["<unk>"])
            else:
                new_sent.append(word)
        new_batch_sents.append(new_sent)
    return new_batch_sents


def evaluate_test(ner_data_loader, path, model_name):
    if args.model_arc == "char_cnn":
	print "Using Char CNN model!"
	model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
	print "Using Char Birnn model!"
	model = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
	print "Using Char Birnn-CNN model!"
	model = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
	print "Using seperate encoders for embedding and features (cnn and birnn char)!"
	model = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
	print "Using seperate encoders for embedding and features (cnn char)!"
	model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
	raise NotImplementedError
    model.load()
    acc, precision, recall, f1 = evaluate(ner_data_loader, path, model, model_name, True)
    return acc, precision, recall, f1

def test_on_full_setE(ner_data_loader, args, data):

    if args.model_arc == "char_cnn":
        print "Using Char CNN model!"
        model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
        print "Using Char Birnn model!"
        model = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
        print "Using Char Birnn-CNN model!"
        model = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
        print "Using seperate encoders for embedding and features (cnn and birnn char)!"
        model = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
        print "Using seperate encoders for embedding and features (cnn char)!"
        model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
        raise NotImplementedError

    model.load()
    # data = (sents, char_sents, discrete_features, origin_sents, bc_feats)
    if args.valid_using_split:
        acc, precision, recall, f1 = evaluate_lr_splitHashtag(ner_data_loader, model,
                                                              "best_" + args.model_name, args.score_file,
                                                              args.setEconll, args, data)
    else:
        acc, precision, recall, f1 = evaluate_lr(ner_data_loader, model, "best_" + args.model_name,
                                                 args.score_file, args.setEconll, data)
    return acc, precision, recall, f1

def main(args):
    prefix = args.model_name + "_" + str(uid)
    print "PREFIX: ", prefix
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    best_output_fname = "../eval/best_%s_darpa_output.conll" % (prefix)
    ner_data_loader = NER_DataLoader(args)

    print ner_data_loader.id_to_tag

    if not args.data_aug:
        sents, char_sents, tgt_tags, discrete_features, bc_features = ner_data_loader.get_data_set(args.train_path, args.lang)
    else:
        sents_tgt, char_sents_tgt, tags_tgt, dfs_tgt, bc_feats_tgt = ner_data_loader.get_data_set(args.tgt_lang_train_path, args.lang)
        sents_aug, char_sents_aug, tags_aug, dfs_aug, bc_feats_aug = ner_data_loader.get_data_set(args.aug_lang_train_path, args.aug_lang)
        sents, char_sents, tgt_tags, discrete_features, bc_features = sents_tgt+sents_aug, char_sents_tgt+char_sents_aug, tags_tgt+tags_aug, dfs_tgt+dfs_aug, bc_feats_tgt+bc_feats_aug

    data_test = ner_data_loader.get_lr_test(args.test_path, args.lang)
    if not args.valid_on_full:
        data_valid = ner_data_loader.get_lr_test(args.dev_path, args.lang)
    else:
        data_valid = data_test
    # print ner_data_loader.char_to_id
    print "Data set size (train): ", len(sents)
    print("Number of discrete features: ", ner_data_loader.num_feats)
    epoch = bad_counter = updates = tot_example = cum_loss = 0
    patience = 30

    display_freq = 100
    valid_freq = args.valid_freq
    batch_size = args.batch_size

    if args.model_arc == "char_cnn":
        print "Using Char CNN model!"
        model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
        print "Using Char Birnn model!"
        model = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
        print "Using Char Birnn-CNN model!"
        model = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
        print "Using seperate encoders for embedding and features (cnn and birnn char)!"
        model = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
        print "Using seperate encoders for embedding and features (cnn char)!"
        model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
        raise NotImplementedError

    # model = debug_vanilla_NER_CRF_model(args, ner_data_loader)
    inital_lr = args.init_lr
    trainer = dy.MomentumSGDTrainer(model.model, inital_lr, 0.9)

    def _check_batch_token(batch, id_to_vocab):
        for line in batch:
            print [id_to_vocab[i] for i in line]

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print [u" ".join([id_to_vocab[c] for c in w]) for w in line]

    lr_decay = args.decay_rate

    decay_patience = 10
    decay_num = 0
    valid_history = []
    best_results = [0.0, 0.0, 0.0, 0.0]
    while epoch <= args.tot_epochs:
        for b_sents, b_char_sents, b_ner_tags, b_feats, b_bc_feats in make_bucket_batches(
                zip(sents, char_sents, tgt_tags, discrete_features, bc_features), batch_size):
            dy.renew_cg()

            if args.replace_unk_rate > 0.0:
                b_sents = replace_singletons(ner_data_loader, b_sents, args.replace_unk_rate)
            # _check_batch_token(b_sents, ner_data_loader.id_to_word)
            # _check_batch_token(b_ner_tags, ner_data_loader.id_to_tag)
            # _check_batch_char(b_char_sents, ner_data_loader.id_to_char)
            loss = model.cal_loss(b_sents, b_char_sents, b_ner_tags, b_feats, b_bc_feats, training=True)
            loss_val = loss.value()
            cum_loss += loss_val * len(b_sents)
            tot_example += len(b_sents)

            # cum_loss += loss_val * len(b_sents) * len(b_sents[0])
            # tot_example += len(b_sents) * len(b_sents[0])

            updates += 1
            loss.backward()
            trainer.update()

            if updates % display_freq == 0:
                # print("avg sum score = %f, avg sent score = %f" % (sum_s.value(), sent_s.value()))
                print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (epoch, updates, loss_val, cum_loss*1.0/tot_example))
            if updates % valid_freq == 0:
                if not args.isLr:
                    acc, precision, recall, f1 = evaluate(ner_data_loader, args.test_path, model, args.model_name,False)
                else:
                    if args.valid_on_full:
                        acc, precision, recall, f1 = evaluate_lr(ner_data_loader, model, args.model_name, args.score_file, args.setEconll, data_valid)
                    else:
                        acc, precision, recall, f1 = evaluate_lr(ner_data_loader, model, args.model_name, args.score_file_10, args.setEconll_10, data_valid)
                    results = [acc, precision, recall, f1]
                    print("Current validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(results))

                if len(valid_history) == 0 or f1 > max(valid_history):
                    bad_counter = 0
                    best_results = [acc, precision, recall, f1]
                    if updates > 2000:
                        print("Saving the best model so far.......")
                        model.save()
                    if args.isLr:
                        os.system("cp %s %s" % (final_darpa_output_fname, best_output_fname))
                else:
                    bad_counter += 1
                    if args.lr_decay and bad_counter >= 5 and os.path.exists(args.save_to_path):
                        bad_counter = 0
                        model.load()

                        print("Epoch = %d, Learning Rate = %f." % (epoch, inital_lr / (1 + epoch * lr_decay)))
                        trainer = dy.MomentumSGDTrainer(model.model, inital_lr / (1 + epoch * lr_decay))

                        # inital_lr = inital_lr * 0.5
                        # decay_num += 1
                        # print("Epoch = %d, Learning Rate = %f." % (epoch, inital_lr))
                        # trainer = dy.MomentumSGDTrainer(model.model, inital_lr)

                if bad_counter > patience:
                # if decay_num > decay_patience:
                    print("Early stop!")
                    print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                    #Test on full SetE
                    if args.isLr:
			acc, precision, recall, f1 = test_on_full_setE(ner_data_loader, args, data_test)
			results = [acc, precision, recall, f1]
			print("Test Result: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(results))

			# post processing
			post_process(args, best_output_fname)
		    else:
			acc,precision,recall, f1 = evaluate_test(ner_data_loader, args.test_path, args.model_name)
			results = [acc, precision, recall, f1]
			print("Test Result: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(results))
			exit(0)
                valid_history.append(f1)
        epoch += 1

    # Test on full SetE
    if args.isLr:
	acc, precision, recall, f1 = test_on_full_setE(ner_data_loader, args, data_test)
	results = [acc, precision, recall, f1]
	print("Test Result: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(results))
	# post processing
	post_process(args, best_output_fname)
    else:
	acc,precision,recall, f1 = evaluate_test(ner_data_loader, args.test_path, args.model_name)
	results = [acc, precision, recall, f1]
	print("Test Result: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(results))
    print("All Epochs done.")
    #print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))


def post_process(args, pred_file):
    fout_name = "../eval/post_" + pred_file.split('/')[-1]
    fout_conll_name = "../eval/post_conll_" + pred_file.split('/')[-1]
    post_score_file = "../eval/post_%s_score_file" % (args.model_name + "_" + str(uid))
    # Currently only support one lookup file
    lookup_file = None if args.lookup_file is None else {"Gen": args.lookup_file}
    post_processing(pred_file, args.setEconll, args.author_file, fout_name, lookup_files=lookup_file,
                    label_propagate=args.label_prop, conf_num=args.confidence_num, gold_file_path=args.gold_setE_path,
                    most_freq_num=args.freq_ngram, fout_conll_name=fout_conll_name)
    print("Score on the post processed file: ")
    os.system("bash %s ../eval/%s %s" % (args.score_file, fout_name, post_score_file))
    with codecs.open(post_score_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec=float(columns[-4])
                recall =float(columns[-3])
                f1 = float(columns[-2])
                break
    print("prec=%f, recall=%f, f1=%f" % (prec, recall, f1))


def test_with_two_models(args):
    # This function is specific for oromo.
    '''We should add --train_lower_case_oromo or --oromo_normalize if the combined model is lower case model or normalized'''
    ner_data_loader = NER_DataLoader(args)
    ner_data_loader_special_normal = NER_DataLoader(args, special_normal=True)
    _, _, _, _, _ = ner_data_loader.get_data_set(args.train_path, args.lang)
    _, _, _, _, _ = ner_data_loader_special_normal.get_data_set(args.train_path, args.lang)
    combine_data_loader = Dataloader_Combine(args,
                                             ner_data_loader_special_normal.word_to_id,
                                             ner_data_loader.word_to_id,
                                             ner_data_loader.char_to_id,
                                             ner_data_loader_special_normal.brown_cluster_dicts,
                                             ner_data_loader.brown_cluster_dicts)
    assert args.load_from_path is not None and args.lower_case_model_path is not None, "Path to the saved models are not provided!"

    if args.model_arc == "char_cnn":
        print "Using Char CNN model!"
        model = vanilla_NER_CRF_model(args, ner_data_loader_special_normal)
        model_lower = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
        print "Using Char Birnn model!"
        model = BiRNN_CRF_model(args, ner_data_loader_special_normal)
        model_lower = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
        print "Using Char Birnn-CNN model!"
        model = CNN_BiRNN_CRF_model(args, ner_data_loader_special_normal)
        model_lower = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
        print "Using seperate encoders for embedding and features (cnn and birnn char)!"
        model = Sep_Encoder_CRF_model(args, ner_data_loader_special_normal)
        model_lower = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
        print "Using seperate encoders for embedding and features (cnn char)!"
        model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader_special_normal)
        model_lower = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
        raise NotImplementedError

    model.load()
    model_lower.load(args.lower_case_model_path)

    sents, char_sents, discrete_features, bc_feats, origin_sents, doc_ids = combine_data_loader.get_lr_test_setE(args.setEconll, args.lang)

    print "Evaluation data size: ", len(sents)
    print("Number of discrete features: ", ner_data_loader.num_feats)
    prefix = args.model_name + "_" + str(uid)
    predictions = []
    i = 0

    predict_with_lower = 0
    for sent, char_sent, discrete_feature, bc_feat, doc_id in zip(sents, char_sents, discrete_features, bc_feats, doc_ids):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]

        if doc_id == "SN":
            best_score, best_path = model_lower.eval(sent, char_sent, discrete_feature, bc_feat, training=False)
            predict_with_lower += 1
        else:
            best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False)
        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    print "%d sents in setE are predicted by the combined model!" % predict_with_lower

    pred_output_fname = "../eval/%s_pred_output.conll" % (prefix)
    with codecs.open(pred_output_fname, "w", "utf-8") as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                if p not in ner_data_loader.id_to_tag:
                    print "ERROR: Predicted tag not found in the id_to_tag dict, the id is: ", p
                    p = 0
                fout.write(word + "\tNNP\tNP\t" + ner_data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (prefix)
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    scoring_file = "../eval/%s_score_file" % (prefix)
    run_program(pred_output_fname, pred_darpa_output_fname, args.setEconll)
    # ../helper_files/
    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)
    if args.valid_using_split:
        os.system("python ../scripts/fix_char_offsets.py --edl_file ../eval/%s --original_LTF_dir ../helper_files/%s/original  --split_hashtag_dir ../helper_files/%s/split_all_hashtags_v2 > ../eval/%s" % (final_darpa_output_fname, args.lang, args.lang, final_darpa_output_fname))
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))
    else:
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))

    prec = 0
    recall = 0
    f1 = 0
    with codecs.open(scoring_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec = float(columns[-4])
                recall = float(columns[-3])
                f1 = float(columns[-2])
                break

    print("Precison=%f, Recall=%f, F1=%F." % (prec, recall, f1))

    post_process(args, final_darpa_output_fname)

def evaluate_test_model(args):
    ner_data_loader = NER_DataLoader(args)
    _, _, _, _, _ = ner_data_loader.get_data_set(args.train_path, args.lang)
    if args.model_arc == "char_cnn":
	print "Using Char CNN model!"
	model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
	print "Using Char Birnn model!"
	model = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
	print "Using Char Birnn-CNN model!"
	model = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
	print "Using seperate encoders for embedding and features (cnn and birnn char)!"
	model = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
	print "Using seperate encoders for embedding and features (cnn char)!"
	model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
	raise NotImplementedError
    evaluate_test(ner_data_loader, args.test_path, args.model_name)


def test_single_model(args):
    ner_data_loader = NER_DataLoader(args)
    # ugly: get discrete number features
    _, _, _, _, _ = ner_data_loader.get_data_set(args.train_path, args.lang)
    if args.model_arc == "char_cnn":
        print "Using Char CNN model!"
        model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
        print "Using Char Birnn model!"
        model = BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn_cnn":
        print "Using Char Birnn-CNN model!"
        model = CNN_BiRNN_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep":
        print "Using seperate encoders for embedding and features (cnn and birnn char)!"
        model = Sep_Encoder_CRF_model(args, ner_data_loader)
    elif args.model_arc == "sep_cnn_only":
        print "Using seperate encoders for embedding and features (cnn char)!"
        model = Sep_CNN_Encoder_CRF_model(args, ner_data_loader)
    else:
        raise NotImplementedError

    model.load()

    sents, char_sents, discrete_features, origin_sents, bc_feats = ner_data_loader.get_lr_test(
        args.test_path, args.lang)

    print "Evaluation data size: ", len(sents)
    print("Number of discrete features: ", ner_data_loader.num_feats)
    prefix = args.model_name + "_" + str(uid)
    predictions = []
    i = 0

    for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]

        best_score, best_path = model.eval(sent, char_sent, discrete_feature, bc_feat, training=False)
        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.conll" % (prefix)
    with codecs.open(pred_output_fname, "w", "utf-8") as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                if p not in ner_data_loader.id_to_tag:
                    print "ERROR: Predicted tag not found in the id_to_tag dict, the id is: ", p
                    p = 0
                fout.write(word + "\tNNP\tNP\t" + ner_data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (prefix)
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    scoring_file = "../eval/%s_score_file" % (prefix)
    run_program(pred_output_fname, pred_darpa_output_fname, args.setEconll)

    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)
    if args.valid_using_split:
        os.system("python ../scripts/fix_char_offsets.py --edl_file ../eval/%s --original_LTF_dir ../helper_files/%s/original  --split_hashtag_dir ../helper_files/%s/split_all_hashtags_v2 > ../eval/%s" % (final_darpa_output_fname, args.lang, args.lang, final_darpa_output_fname))
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))
    else:
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))

    prec = 0
    recall = 0
    f1 = 0
    with codecs.open(scoring_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec = float(columns[-4])
                recall = float(columns[-3])
                f1 = float(columns[-2])
                break

    print("Precison=%f, Recall=%f, F1=%F." % (prec, recall, f1))

    post_process(args, final_darpa_output_fname)


def ensemble_test_single_model(args):
    # Note: when train ensemble, provide both full_data_path and train_path
    # when test ensemble, provide just the train_path as the full_data_path
    ner_data_loader = NER_DataLoader(args)
    # ugly: get discrete number features
    _, _, _, _, _ = ner_data_loader.get_data_set(args.train_path, args.lang)
    paths = []
    with open(args.ensemble_model_paths) as fin:
        for line in fin:
            paths.append(line.strip())
    models = []

    for i, path in enumerate(paths):
        if args.model_arc == "char_cnn":
            print "Using Char CNN model!"
            models.append(vanilla_NER_CRF_model(args, ner_data_loader))
        elif args.model_arc == "char_birnn":
            print "Using Char Birnn model!"
            models.append(BiRNN_CRF_model(args, ner_data_loader))
        elif args.model_arc == "char_birnn_cnn":
            print "Using Char Birnn-CNN model!"
            models.append(CNN_BiRNN_CRF_model(args, ner_data_loader))
        elif args.model_arc == "sep":
            print "Using seperate encoders for embedding and features (cnn and birnn char)!"
            models.append(Sep_Encoder_CRF_model(args, ner_data_loader))
        elif args.model_arc == "sep_cnn_only":
            print "Using seperate encoders for embedding and features (cnn char)!"
            models.append(Sep_CNN_Encoder_CRF_model(args, ner_data_loader))
        else:
            raise NotImplementedError
        models[i].load(path)

    sents, char_sents, discrete_features, origin_sents, bc_feats = ner_data_loader.get_lr_test(
        args.test_path, args.lang)

    print "Evaluation data size: ", len(sents)
    print("Number of discrete features: ", ner_data_loader.num_feats)
    prefix = args.model_name + "_" + str(uid)
    predictions = []
    i = 0
    print "Start ensembling using %d models!" % len(models)

    for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
        dy.renew_cg()
        sent, char_sent, discrete_feature, bc_feat = [sent], [char_sent], [discrete_feature], [bc_feat]
        tag_scores = []
        transit_scores = []
        for model in models:
            trs, ts = model.eval_scores(sent, char_sent, discrete_feature, bc_feat, training=False)
            tag_scores.append(ts)
            transit_scores.append(trs)

        best_score, best_path = ensemble_viterbi_decoding(tag_scores, transit_scores, len(ner_data_loader.tag_to_id))
        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.conll" % (prefix)
    with codecs.open(pred_output_fname, "w", "utf-8") as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                if p not in ner_data_loader.id_to_tag:
                    print "ERROR: Predicted tag not found in the id_to_tag dict, the id is: ", p
                    p = 0
                fout.write(word + "\tNNP\tNP\t" + ner_data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (prefix)
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (prefix)
    scoring_file = "../eval/%s_score_file" % (prefix)
    run_program(pred_output_fname, pred_darpa_output_fname, args.setEconll)

    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)
    if args.valid_using_split:
        os.system("python ../scripts/fix_char_offsets.py --edl_file ../eval/%s --original_LTF_dir ../helper_files/%s/original  --split_hashtag_dir ../helper_files/%s/split_all_hashtags_v2 > ../eval/%s" % (final_darpa_output_fname, args.lang, args.lang, final_darpa_output_fname))
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))
    else:
        os.system("bash %s ../eval/%s %s" % (args.score_file, final_darpa_output_fname, scoring_file))

    prec = 0
    recall = 0
    f1 = 0
    with codecs.open(scoring_file, 'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec = float(columns[-4])
                recall = float(columns[-3])
                f1 = float(columns[-2])
                break

    print("Precison=%f, Recall=%f, F1=%F." % (prec, recall, f1))

    post_process(args, final_darpa_output_fname)


def init_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", default=5783287, type=int)

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--lang", default="english", help="the target language")
    parser.add_argument("--train_ensemble", default=False, action="store_true")
    parser.add_argument("--full_data_path", type=str, default=None, help="when train_ensemble is true, this one is the full data path from which to load vocabulary.")
    parser.add_argument("--train_path", default="../datasets/english/eng.train.bio.conll", type=str)
    # parser.add_argument("--train_path", default="../datasets/english/debug_train.bio", type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--save_to_path", default="../saved_models/")
    parser.add_argument("--load_from_path", default=None)

    # oromo specific argument
    # No matter orm_norm or orm_lower, the char representation is from the original word
    parser.add_argument("--lower_case_model_path", type=str, default=None)
    parser.add_argument("--train_lowercase_oromo", default=False, action="store_true")
    parser.add_argument("--oromo_normalize", default=False, action="store_true", help="if train lowercase model, not sure if norm also helps, this would loss a lot of information")

    parser.add_argument("--model_arc", default="char_cnn", choices=["char_cnn", "char_birnn", "char_birnn_cnn", "sep", "sep_cnn_only"], type=str)
    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=50, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", choices=['lstm', 'gru'], type=str)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--char_hidden_dim", default=25, type=int, help="char level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)

    parser.add_argument("--replace_unk_rate", default=0.0, type=float, help="uses when not all words in the test data is covered by the pretrained embedding")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--map_pretrain", default=False, action="store_true")
    parser.add_argument("--map_dim", default=100, type=int)
    parser.add_argument("--pretrain_fix", default=False, action="store_true")

    parser.add_argument("--output_dropout_rate", default=0.5, type=float, help="dropout applied to the output of birnn before crf")
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float, help="dropout applied to the input of token-level birnn")
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--init_lr", default=0.015, type=float)
    parser.add_argument("--lr_decay", default=False, action="store_true")
    parser.add_argument("--decay_rate", default=0.05, action="store", type=float)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--data_aug", default=False, action="store_true", help="If use data_aug, the train_path should be the combined training file")
    parser.add_argument("--aug_lang", default="english", help="the language to augment the dataset")
    parser.add_argument("--aug_lang_train_path", default=None, type=str)
    parser.add_argument("--tgt_lang_train_path", default="../datasets/english/eng.train.bio.conll", type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)

    parser.add_argument("--feature_birnn_hidden_dim", default=50, type=int, action="store")

    parser.add_argument("--use_discrete_features", default=False, action="store_true", help="David's indicator features")
    parser.add_argument("--feature_dim", type=int, default=10, help="dimension of discrete features")
    parser.add_argument("--use_brown_cluster", default=False, action="store_true")
    parser.add_argument("--brown_cluster_path", action="store", type=str, help="path to the brown cluster features")
    parser.add_argument("--brown_cluster_num", default=500, type=int, action="store")
    parser.add_argument("--brown_cluster_dim", default=30, type=int, action="store")
    parser.add_argument("--use_gazatter", default=False, action="store_true")
    parser.add_argument("--use_morph", default=False, action="store_true")

    # CRF decoding
    parser.add_argument("--interp_crf_score", default=False, action="store_true", help="if True, interpolate between the transition and emission score.")
    # post process arguments
    parser.add_argument("--label_prop", default=False, action="store_true")
    parser.add_argument("--confidence_num", default=2, type=str)
    parser.add_argument("--author_file", default=None, type=str)
    parser.add_argument("--lookup_file", default=None, type=str)
    parser.add_argument("--freq_ngram", default=20, type=int)

    parser.add_argument("--isLr", default=False, action="store_true")
    parser.add_argument("--valid_on_full", default=False, action="store_true")
    parser.add_argument("--valid_using_split", default=False, action="store_true")
    parser.add_argument("--setEconll", type=str, default=None, help="path to the full setE conll file")
    parser.add_argument("--setEconll_10", type=str, default=None, help="path to the 10% setE conll file")
    parser.add_argument("--score_file", type=str, default=None,help="path to the scoring file for full setE conll file")
    parser.add_argument("--score_file_10", type=str, default=None, help="path to the scoring file for 10% setE conll file")

    parser.add_argument("--gold_setE_path", type=str, default="../ner_score/")
    # Use trained model to test
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test_2", "test_1", "ensemble", "pred_ensemble",],
                        help="test_1: use one model; test_2: use lower case model and normal model to test oromo; "
                             "ensemble: CRF ensemble; pred_ensemble: ensemble prediction results")
    parser.add_argument("--ensemble_model_paths", type=str, help="each line in this file is the path to one model")
    args = parser.parse_args()

    # We are not using uuid to make a unique time stamp, since I thought there is no need to do so when we specify a good model_name.

    # If use score_10pct.sh, put the setE_10pct.txt as the dev_path
    # If use valid_using_split, set the test_path and setEconll to be the splitted version, this is used for full setE testing
    if args.train_ensemble:
        # model_name = ens_1_ + original
        # set dynet seed manually
        ens_no = int(args.model_name.split("_")[1])
        # dyparams = dy.DynetParams()
        # dyparams.set_random_seed(ens_no + 5783287)
        # dyparams.init()

        import dynet_config
        dynet_config.set(random_seed=ens_no + 5783290)
        # if args.cuda:
        #     dynet_config.set_gpu()

        # args.train_path = args.train_path.split(".")[0] + "_" + str(ens_no) + ".conll"

    if args.full_data_path is None:
        args.full_data_path = args.train_path
    args.save_to_path = args.save_to_path + args.model_name + ".model"
    args.gold_setE_path = args.gold_setE_path + args.lang + "_setE_edl.tac"
    print args
    return args

args = init_config()
from models.model_builder import *
import os
import uuid
from utils.Convert_to_darpa_xml import *
# from dataloaders.dataloader_unicode import *
from dataloaders.data_loader import *
from utils.Convert_Output_Darpa import *
from utils.post_process import post_processing
uid = uuid.uuid4().get_hex()[:6]

if __name__ == "__main__":
    # args = init_config()
    if args.mode == "train":
        args.load_from_path = args.save_to_path
	print "in evalEng"
	exit(0)
        main(args)
    elif args.mode == "test_1":
        if args.isLr:
	    test_single_model(args)
	else:
	    evaluate_test_model(args)
    elif args.mode == "test_2":
        test_with_two_models(args)
    elif args.mode == "ensemble":
        ensemble_test_single_model(args)
    else:
        raise NotImplementedError
