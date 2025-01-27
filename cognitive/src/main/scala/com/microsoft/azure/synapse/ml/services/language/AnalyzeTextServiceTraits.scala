package com.microsoft.azure.synapse.ml.services.language

import com.microsoft.azure.synapse.ml.param.ServiceParam
import com.microsoft.azure.synapse.ml.services.HasServiceParams
import com.microsoft.azure.synapse.ml.services.language.ATJSONFormat._
import com.microsoft.azure.synapse.ml.services.text.TADocument
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Row
import spray.json.DefaultJsonProtocol._
import spray.json.enrichAny

object AnalysisTaskKind extends Enumeration {
  type AnalysisTaskKind = Value
  val SentimentAnalysis,
  EntityRecognition,
  PiiEntityRecognition,
  KeyPhraseExtraction,
  LanguageDetection,
  EntityLinking = Value
}

private[language] trait AnalyisisInputCreator {
  protected def createAnalysisDocument[T, V](validTexts: Seq[String],
                                             optionalParam: Option[Seq[String]],
                                             inputContructor: (Option[String], String, String) => T,
                                             documentConstructor: (Seq[T]) => V): V = {
    val params = optionalParam.getOrElse(Seq.fill(validTexts.length)(""))
    val validParams = (if (params.length == 1) {
      Seq.fill(validTexts.length)(params.head)
    } else {
      params
    }).map(Option(_))

    val documents: Seq[T] = validTexts.zipWithIndex.map {
      case (text, i) => inputContructor(validParams(i), i.toString, text)
    }
    documentConstructor(documents)
  }
}


trait AnalyzeTextServiceBaseParameters extends HasServiceParams {
  // modelVersion related fields and methods
  val modelVersion = new ServiceParam[String](
    this, name = "modelVersion", "Version of the model")

  def setModelVersion(v: String): this.type = setScalarParam(modelVersion, v)

  def setModelVersionCol(v: String): this.type = setVectorParam(modelVersion, v)

  def getModelVersion: String = getScalarParam(modelVersion)

  def getModelVersionCol: String = getVectorParam(modelVersion)

  // loggingOptOut related fields and methods
  val loggingOptOut = new ServiceParam[Boolean](this, "loggingOptOut", "loggingOptOut for task")

  def setLoggingOptOut(v: Boolean): this.type = setScalarParam(loggingOptOut, v)

  def getLoggingOptOut: Boolean = getScalarParam(loggingOptOut)

  def setLoggingOptOutCol(v: String): this.type = setVectorParam(loggingOptOut, v)

  def getLoggingOptOutCol: String = getVectorParam(loggingOptOut)


  // stringIndexType related fields and methods
  val stringIndexType = new ServiceParam[String](this,
                                                 "stringIndexType",
                                                 "Specifies the method used to interpret string offsets. Defaults to " +
                                                   "Text Elements (Graphemes) according to Unicode v8.0.0. For " +
                                                   "additional information see https://aka.ms/text-analytics-offsets",
                                                 isValid = {
                                                   case Left(s) => Set("TextElements_v8", "UnicodeCodePoint",
                                                                       "Utf16CodeUnit")(s)
                                                   case _ => true
                                                 })

  def setStringIndexType(v: String): this.type = setScalarParam(stringIndexType, v)

  def getStringIndexType: String = getScalarParam(stringIndexType)

  def setStringIndexTypeCol(v: String): this.type = setVectorParam(stringIndexType, v)

  def getStringIndexTypeCol: String = getVectorParam(stringIndexType)


  // kind related fields and methods
  // We don't support setKindCol here because output schemas for different kind are different
  val kind = new Param[String](
    this,
    "kind",
    "Enumeration of supported Text Analysis tasks",
    isValid = AnalysisTaskKind.values.map(_.toString).contains(_)
    )

  def setKind(v: String): this.type = set(kind, v)

  def getKind: String = $(kind)

  private[language] def getTypedKind: AnalysisTaskKind.Value = AnalysisTaskKind.withName(getKind)

  // showStats related fields and methods
  val showStats = new ServiceParam[Boolean](
    this, name = "showStats", "Whether to include detailed statistics in the response",
    isURLParam = true)

  def setShowStats(v: Boolean): this.type = setScalarParam(showStats, v)

  def getShowStats: Boolean = getScalarParam(showStats)
}


trait HandleSentimentAnalysis extends HasServiceParams with AnalyisisInputCreator {
  val opinionMining = new ServiceParam[Boolean](
    this,
    name = "opinionMining",
    isRequired = false,
    doc = "Whether to use opinion mining in the request or not."
    )

  def getOpinionMining: Boolean = getScalarParam(opinionMining)

  def setOpinionMining(value: Boolean): this.type = setScalarParam(opinionMining, value)

  def getOpinionMiningCol: String = getVectorParam(opinionMining)

  def setOpinionMiningCol(value: String): this.type = setVectorParam(opinionMining, value)

  def createSentimentAnalysisRequest(row: Row,
                                     texts: Seq[String],
                                     languages: Option[Seq[String]],
                                     modelVersion: String,
                                     stringIndexType: String,
                                     loggingOptOut: Boolean): String = {
    val analyisisParam = createAnalysisDocument(texts,
                                                languages,
                                                TADocument.apply,
                                                MultiLanguageAnalysisInput.apply)
    val sentimentAnalysisTaskParameters = SentimentAnalysisTaskParameters(loggingOptOut,
                                                                          modelVersion,
                                                                          getOpinionMining,
                                                                          stringIndexType)

    SentimentAnalysisInput(analyisisParam,
                           sentimentAnalysisTaskParameters,
                           AnalysisTaskKind.SentimentAnalysis.toString
                           ).toJson.compactPrint
  }

  def createSentimentAnalysisLRORequest(row: Row,
                                        texts: Seq[String],
                                        languages: Option[Seq[String]],
                                        modelVersion: String,
                                        stringIndexType: String,
                                        loggingOptOut: Boolean): String = {
    val analyisisParam = createAnalysisDocument(texts,
                                                languages,
                                                TADocument.apply,
                                                MultiLanguageAnalysisInput.apply)
    val sentimentAnalysisTaskParameters = SentimentAnalysisTaskParameters(loggingOptOut,
                                                                          modelVersion,
                                                                          getOpinionMining,
                                                                          stringIndexType)

    val sentimentAnalisysLroTask = SentimentAnalysisLROTask(sentimentAnalysisTaskParameters,
                                                            Some(AnalysisTaskKind.SentimentAnalysis.toString),
                                                            AnalysisTaskKind.SentimentAnalysis.toString)
    val request = SentimentAnalysisJobsInput(Some("Job"), analyisisParam, Seq(sentimentAnalisysLroTask))
    request.toJson.compactPrint
  }
}
